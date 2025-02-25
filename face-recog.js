let imageBlobs = [];
let model;

async function loadBlazeFace() {
    model = await blazeface.load();
    console.log("BlazeFace Model Loaded");
}

async function captureFace(canvas, ctx, video) {
    if (!model) return;

    const predictions = await model.estimateFaces(video, false);
    
    if (predictions.length > 0) {
        const face = predictions[0];  // Take the first detected face

        const x = face.topLeft[0];
        const y = face.topLeft[1];
        const width = face.bottomRight[0] - x;
        const height = face.bottomRight[1] - y;

        canvas.width = 200; 
        canvas.height = 200;

        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(video, x, y, width, height, 0, 0, canvas.width, canvas.height);

        // Convert canvas to Blob and store
        canvas.toBlob(blob => {
            if (blob) {
                imageBlobs.push(blob);
                console.log(`Captured Face ${imageBlobs.length}/3`);

                // Once 3 images are captured, send them for recognition
                if (imageBlobs.length === 3) {
                    console.log("3 face captures reached. Starting recognition...");
                    sendImagesToFlask(imageBlobs);
                    imageBlobs = []; // Clear array for next batch
                }
            }
        }, "image/jpeg", 0.9);
    }
}

function sendImagesToFlask(images) {
    const formData = new FormData();
    images.forEach((blob, index) => {
        formData.append("face", blob, `face_${index}.jpg`);
    });

    fetch("http://192.168.100.88:6600/upload", {
        method: "POST",
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            // If response is not OK, attempt to extract JSON error message
            return response.json().then(err => {
                throw new Error(err.error || `Server returned ${response.status}`);
            }).catch(() => {
                throw new Error(`Server returned ${response.status} - Unable to parse JSON`);
            });
        }
        return response.json();
    })
    .then(data => {
        console.log("Server Response:", data);
        if (data.message) {
            console.log(`Recognition Successful: ${data.recognized_id}`);
        } else {
            console.log(`Recognition Failed: ${data.error}`);
        }
    })
    .catch(error => {
        console.error("Upload error:", error);
        console.log(`Error uploading images: ${error.message}`);
    });
}

async function startVideo() {
    const video = document.getElementById("video");
    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");

    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
        video.play();
    } catch (error) {
        console.error("Error accessing camera:", error);
        console.log("Could not access camera. Please allow permissions.");
        return;
    }

    await loadBlazeFace();

    setInterval(() => {
        captureFace(canvas, ctx, video);
    }, 500); // Capture every 500ms
}

// Start video streaming and face detection
startVideo().catch(console.error);
