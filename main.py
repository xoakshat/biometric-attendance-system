import face_recognition
import cv2
import numpy as np
import os
import json
from datetime import datetime
from appwrite.client import Client
from appwrite.services.databases import Databases
from appwrite.query import Query

# ─────────────────────────────────────────────
#  APPWRITE CONFIGURATION — fill these in
# ─────────────────────────────────────────────
APPWRITE_ENDPOINT  = "https://sgp.cloud.appwrite.io/v1"
APPWRITE_PROJECT   = "biometricsrmu"        # Your project ID
APPWRITE_API_KEY   = "standard_786794388045218ce4c19ebdc3c19eba6c5f767af7b8ef048df7c26d52f548f766af9156ca876c53017e14717a03e78dc6c1ba48c93187dbd1cea946a8ba6d4e9d9d6a66276bc8c2fd9dff2ec9d8d2845f48f8b37146a41363ee8fa71e4fcc582ad6b8bad007f438eaab3575b5bacbdc493519057a5e0f0ddf8946ec3adba7d0"         # Settings → API Keys → create one
DATABASE_ID        = "69e094c9000e2ef85c2a"      # attendanceDB
COLLECTION_ID      = "students"                  # students collection ID (check Appwrite)
PHOTOS_DIR         = "photos"                    # folder with student photos
# ─────────────────────────────────────────────

# Init Appwrite client
client = Client()
client.set_endpoint(APPWRITE_ENDPOINT)
client.set_project(APPWRITE_PROJECT)
client.set_key(APPWRITE_API_KEY)
databases = Databases(client)


def load_known_faces():
    """Load all student photos and encode their faces."""
    known_encodings = []
    known_names     = []
    known_rollnos   = []

    for filename in os.listdir(PHOTOS_DIR):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        filepath = os.path.join(PHOTOS_DIR, filename)
        image    = face_recognition.load_image_file(filepath)
        encs     = face_recognition.face_encodings(image)

        if not encs:
            print(f"[WARN] No face found in {filename}, skipping.")
            continue

        # filename format: firstname_lastname.jpg  OR  name_rollno.jpg
        base = os.path.splitext(filename)[0]          # e.g. "abhay_singh_arya"
        name = base.replace("_", " ").title()         # "Abhay Singh Arya"

        known_encodings.append(encs[0])
        known_names.append(name)
        known_rollnos.append(base)                    # used as unique key

        print(f"[INFO] Loaded: {name}")

    return known_encodings, known_names, known_rollnos


def register_students_in_appwrite(known_names, known_rollnos):
    """
    Register students in Appwrite if they don't exist yet.
    Run this once to populate the database from your photos folder.
    """
    for name, rollno in zip(known_names, known_rollnos):
        # check if already registered
        try:
            result = databases.list_documents(
                DATABASE_ID, 
                COLLECTION_ID,
                queries=[Query.equal("rollNo", rollno)]
            )
            if result["total"] > 0:
                print(f"[SKIP] {name} already in DB.")
                continue
        except Exception as e:
            print(f"[ERROR] Checking {name}: {e}")
            continue

        # create new document
        try:
            databases.create_document(
                DATABASE_ID,
                COLLECTION_ID,
                "unique()",
                {
                    "name":       name,
                    "rollNo":     rollno,
                    "image":      f"{rollno}.jpg",
                    "attendance": json.dumps([])
                }
            )
            print(f"[OK] Registered: {name}")
        except Exception as e:
            print(f"[ERROR] Registering {name}: {e}")


def mark_attendance(rollno, databases):
    today = datetime.now().strftime("%Y-%m-%d")

    try:
        result = databases.list_documents(
            DATABASE_ID,
            COLLECTION_ID,
            queries=[Query.equal("rollNo", rollno)]
        )

        if result["total"] == 0:
            print(f"[WARN] Student {rollno} not found in DB.")
            return False

        doc = result["documents"][0]
        doc_id = doc["$id"]
        raw = doc.get("attendance", "[]")
        att_list = json.loads(raw) if isinstance(raw, str) else raw

        if today in att_list:
           print(f"[INFO] {rollno} already marked today ({today})")
           return False

        att_list.append(today)

        databases.update_document(
            DATABASE_ID,
            COLLECTION_ID,
            doc_id,
            data={"attendance": json.dumps(att_list)}
        )

        print(f"[✓] Attendance marked for {rollno} on {today}")
        return True

    except Exception as e:
        print(f"[ERROR] Marking attendance for {rollno}: {e}")
        return False


def run_attendance_camera(known_encodings, known_names, known_rollnos):
    """Main loop: open webcam, detect faces, mark attendance."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        return

    print("[INFO] Camera started. Press 'q' to quit.")
    marked_today = set()   # avoid re-marking in same session

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize for speed
        small  = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb    = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        locations  = face_recognition.face_locations(rgb)
        encodings  = face_recognition.face_encodings(rgb, locations)

        for enc, loc in zip(encodings, locations):
            matches   = face_recognition.compare_faces(known_encodings, enc, tolerance=0.5)
            distances = face_recognition.face_distance(known_encodings, enc)
            best      = int(np.argmin(distances)) if len(distances) > 0 else -1

            label  = "Unknown"
            color  = (0, 0, 255)

            if best >= 0 and matches[best]:
                name   = known_names[best]
                rollno = known_rollnos[best]
                label  = name

                if rollno not in marked_today:
                    success = mark_attendance(rollno, databases)
                    if success:
                        marked_today.add(rollno)
                        label = f"{name} ✔"
                        color = (0, 255, 0)
                    else:
                        label = f"{name} (done)"
                        color = (255, 165, 0)
                else:
                    color = (255, 165, 0)

            # Scale back up face location
            top, right, bottom, left = [v * 4 for v in loc]
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 30), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, label, (left + 4, bottom - 8),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

        # Show timestamp
        ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        cv2.putText(frame, ts, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Biometric Attendance System", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print(" Biometric Attendance System")
    print("=" * 50)

    print("\n[1] Loading faces from photos folder...")
    encodings, names, rollnos = load_known_faces()

    if not encodings:
        print("[ERROR] No faces loaded. Add photos to the 'photos' folder.")
        exit(1)

    print(f"\n[2] Registering {len(names)} students in Appwrite...")
    register_students_in_appwrite(names, rollnos)

    print("\n[3] Starting camera for attendance...")
    run_attendance_camera(encodings, names, rollnos)