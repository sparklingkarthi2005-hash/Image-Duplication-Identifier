import os
import io
import zipfile
import cv2
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, send_file, make_response

app = Flask(__name__)

# Configure upload and temp folders
UPLOAD_FOLDER = 'temp_uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def compute_dhash(image, hash_size=8):
    """
    Computes difference hash (dHash) for an image array.
    """
    # Resize image to (hash_size + 1, hash_size)
    resized = cv2.resize(image, (hash_size + 1, hash_size))
    # Compute difference between adjacent pixels
    diff = resized[:, 1:] > resized[:, :-1]
    # Convert boolean array to a bitstring hash
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])

def get_all_rotation_hashes(cv_img):
    """
    Generates dHashes for 0°, 90°, 180°, 270° rotations and horizontal mirror flips.
    """
    hashes = set()
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    
    # 4 Rotations
    for angle in [0, 90, 180, 270]:
        if angle == 0:
            rotated = gray
        elif angle == 90:
            rotated = cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            rotated = cv2.rotate(gray, cv2.ROTATE_180)
        elif angle == 270:
            rotated = cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
        hashes.add(compute_dhash(rotated))
        
        # Horizontal Mirror Flip
        flipped = cv2.flip(rotated, 1)
        hashes.add(compute_dhash(flipped))
        
    return hashes

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_images():
    uploaded_files = request.files.getlist('images')
    if not uploaded_files or uploaded_files[0].filename == '':
        return "No files uploaded", 400

    image_hashes = {}
    seen_hashes = set()
    unique_images = []
    duplicate_reports = []
    duplicates_count = 0

    # Step 1: Read images and perform Rotation/Mirror aware Deduplication
    for file in uploaded_files:
        filename = os.path.basename(file.filename)
        if not filename:
            continue
            
        file_bytes = file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            continue

        # Get all 8 transformations (4 rotations x 2 flips)
        rot_hashes = get_all_rotation_hashes(img)

        # Check if any rotation/flip hash matches an already processed image
        is_duplicate = False
        matched_original = ""
        
        for h in rot_hashes:
            if h in seen_hashes:
                is_duplicate = True
                matched_original = image_hashes[h]
                break

        if is_duplicate:
            duplicates_count += 1
            duplicate_reports.append(f"DUPLICATE REMOVED: '{filename}' (Matches original: '{matched_original}')")
        else:
            # Store primary hash and track unique file
            base_hash = compute_dhash(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            for h in rot_hashes:
                seen_hashes.add(h)
                image_hashes[h] = filename
            
            unique_images.append((filename, file_bytes))

    # Step 2: Build Text Report Content
    report_lines = [
        "===============================================",
        "   ROTATION-AWARE AI DEDUPLICATION REPORT   ",
        "===============================================",
        f"Total Images Processed : {len(uploaded_files)}",
        f"Unique Images Retained : {len(unique_images)}",
        f"Duplicates Found/Removed: {duplicates_count}",
        "-----------------------------------------------\n",
        "DETAILED DUPLICATE LOGS:"
    ]
    if duplicate_reports:
        report_lines.extend(duplicate_reports)
    else:
        report_lines.append("No duplicate or rotated images found in this dataset.")

    report_content = "\n".join(report_lines)

    # Step 3: Create In-Memory ZIP File
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Write unique clean images
        for fname, fbytes in unique_images:
            zip_file.writestr(f"cleaned_dataset/{fname}", fbytes)
            
        # Write report file
        zip_file.writestr("enhanced_deep_scan_duplicates.txt", report_content)

    zip_buffer.seek(0)

    # Step 4: Return Response with Custom Header for History UI
    response = make_response(
        send_file(
            zip_buffer,
            mimetype='application/zip',
            as_attachment=True,
            download_name='cleaned_dataset_with_report.zip'
        )
    )
    # Header used by Frontend JS to render the exact duplicate count in history
    response.headers['X-Duplicate-Count'] = str(duplicates_count)
    return response

if __name__ == '__main__':
    app.run(debug=True, port=5000)