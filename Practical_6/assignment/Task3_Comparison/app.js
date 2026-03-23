const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

const singleText = document.getElementById("single");
const multiText = document.getElementById("multi");

const video = document.createElement("video");

canvas.width = 640;
canvas.height = 480;

// Setup camera
async function setupCamera() {
    const stream = await navigator.mediaDevices.getUserMedia({
        video: true
    });
    video.srcObject = stream;

    return new Promise(resolve => {
        video.onloadedmetadata = () => resolve(video);
    });
}

// Calculate average keypoint accuracy
function getAvgKeypointScore(keypoints) {
    let total = 0;
    keypoints.forEach(k => total += k.score);
    return total / keypoints.length;
}

// Draw skeleton
function drawSkeleton(keypoints, color, width) {
    const pairs = posenet.getAdjacentKeyPoints(keypoints, 0.3);

    pairs.forEach(pair => {
        ctx.beginPath();
        ctx.moveTo(pair[0].position.x, pair[0].position.y);
        ctx.lineTo(pair[1].position.x, pair[1].position.y);
        ctx.strokeStyle = color;
        ctx.lineWidth = width;
        ctx.stroke();
    });
}

async function main() {
    await setupCamera();
    video.play();

    const net = await posenet.load();

    async function detect() {

        // SINGLE POSE
        const singlePose = await net.estimateSinglePose(video, {
            flipHorizontal: true
        });

        // MULTI POSE (FORCED DIFFERENCE)
        const multiPoses = await net.estimateMultiplePoses(video, {
            flipHorizontal: true,
            maxDetections: 5,
            scoreThreshold: 0.7,   // stricter
            nmsRadius: 50          // more suppression
        });

        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Draw video
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        // 🟢 SINGLE (thick)
        drawSkeleton(singlePose.keypoints, "#28a745", 4);

        // 🔴 MULTI (thin)
        multiPoses.forEach(pose => {
            drawSkeleton(pose.keypoints, "#e74c3c", 2);
        });

        // ACCURACY CALCULATION
        const singleAcc = getAvgKeypointScore(singlePose.keypoints);

        let multiAcc = 0;
        multiPoses.forEach(p => {
            multiAcc += getAvgKeypointScore(p.keypoints);
        });
        multiAcc = multiPoses.length ? multiAcc / multiPoses.length : 0;

        singleText.innerText = singleAcc.toFixed(2);
        multiText.innerText = multiAcc.toFixed(2);

        // Label
        ctx.fillStyle = "black";
        ctx.font = "16px Arial";
        ctx.fillText("Green: Single | Red: Multi", 10, 20);

        requestAnimationFrame(detect);
    }

    detect();
}

main();