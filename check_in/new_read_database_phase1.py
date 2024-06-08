import sqlite3
import numpy as np

# Connect to the SQLite database
conn = sqlite3.connect("face_encodings_deepface_2.db")  # Ensure this matches the new database name
cursor = conn.cursor()

# Retrieve all data from the face_encodings table
cursor.execute("SELECT encoding, name, department, class_name, id_code, library FROM face_encodings_deepface")
rows = cursor.fetchall()

# Iterate over the retrieved data
for row in rows:
    encoding_bytes = row[0]
    name = row[1]
    department = row[2]  # New field
    class_name = row[3]  # New field
    id_code = row[4]
    library = row[5]
    
    # Convert the encoding bytes to a NumPy array
    encoding = np.frombuffer(encoding_bytes, dtype=np.float64)  # Ensure dtype matches the encoding type
    
    # Perform actions with the encoding, name, department, class, id_code, sequence_number, and library
    print("Name:", name)
    print("Department:", department)  # New field
    print("Class:", class_name)  # New field
    print("ID Code:", id_code)
    print("Encoding:", encoding)
    print("Library:", library)
    print()

# Close the database connection
conn.close()