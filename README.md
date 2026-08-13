# 🖼️ Multi-Hash AI Image Deduplicator & Dataset Cleaner :-

A high-performance, rotation-aware web application built to detect and eliminate duplicate, rotated, or mirrored images from Computer Vision datasets. Powered by **Python, Flask, OpenCV, and Multi-Hash Algorithms**.

## 📌 Problem Statement :-

When building Machine Learning and Computer Vision models, duplicate or near-duplicate images in the training dataset lead to **data leakage** and **model overfitting**. Manual cleanup is tedious, and simple filename comparisons miss images that are:
* Rotated ($90^\circ, 180^\circ, 270^\circ$)
* Horizontally flipped (mirrored)
* Re-compressed or slightly resized

This application automates the cleanup process using perceptual hashing and spatial transformations.

## ✨ Key Features :-

* 🔍 **Multi-Hash Deep Scanning:** Combines **aHash, dHash, pHash, and wHash** to generate structural fingerprints for each image.
* 🔄 **Rotation & Mirror Detection:** Uses OpenCV matrix transformations to catch duplicates regardless of orientation.
* 📂 **Bulk & Folder Uploads:** Drag-and-drop individual files or entire folder hierarchies directly from the browser.
* ⚡ **Automated Zip Clean-Up:** Automatically isolates unique images and provides a downloadable clean ZIP package.
* 📊 **Audit Logs & History:** Generates detailed text reports (`enhanced_deep_scan_duplicates.txt`) and tracks real-time scan session history.
* 🎨 **Modern Responsive UI:** Dark glassmorphism frontend built for desktop and mobile environments.

## 🛠️ Tech Stack & Dependencies :-

* **Language:** Python 3
* **Backend Framework:** Flask
* **Image Processing & Computer Vision:** OpenCV (`opencv-python-headless`), PIL (`Pillow`), `imagehash`, `numpy`
* **Production Web Server:** Gunicorn
* **Frontend:** HTML5, CSS3, JavaScript (Fetch API)
* **Hosting Platform:** Render.com

## ⚙️ Algorithms Used :-

1. **Difference Hashing (dHash) & Perceptual Hashing (pHash):** Converts images to grayscale, scales them to structural grids, and generates 64-bit binary hashes resistant to scaling and resolution changes.
2. **Spatial Matrix Transformations:** Applies $90^\circ, 180^\circ, 270^\circ$ rotations and horizontal flipping before comparison.
3. **Hamming Distance Thresholding:** Uses bitwise XOR comparison to calculate exact and near-duplicate matching scores.
