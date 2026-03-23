const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

let sCount = 0, sStage = "up";
let hCount = 0, hStage = "down";

// Camera setup
async function setupCamera() {
    const stream = await navigator.mediaDevices.getUserMedia({
        video: true
    });
    video.srcObject = stream;

    return new Promise(resolve => {
        video.onloadedmetadata = () => resolve(video);
    });
}

// Angle calculation
function findAngle(A, B, C) {
    let radians = Math.atan2(C.y - B.y, C.x - B.x) -
                  Math.atan2(A.y - B.y, A.x - B.x);
    let angle = Math.abs(radians * 180.0 / Math.PI);
    if (angle > 180) angle = 360 - angle;
    return angle;
}

// Draw skeleton
function drawSkeleton(keypoints) {
    const pairs = posenet.getAdjacentKeyPoints(keypoints, 0.2);

    pairs.forEach(pair => {
        ctx.beginPath();
        ctx.moveTo(pair[0].position.x, pair[0].position.y);
        ctx.lineTo(pair[1].position.x, pair[1].position.y);
        ctx.strokeStyle = "#28a745";
        ctx.lineWidth = 3;
        ctx.stroke();
    });
}

// Draw keypoints
function drawPoints(keypoints) {
    keypoints.forEach(point => {
        if (point.score > 0.2) {
            ctx.beginPath();
            ctx.arc(point.position.x, point.position.y, 5, 0, 2*Math.PI);
            ctx.fillStyle = "#007bff";
            ctx.fill();
        }
    });
}

async function main() {
    await setupCamera();
    video.play();

    const net = await posenet.load();

    async function detect() {
        const pose = await net.estimateSinglePose(video, {
            flipHorizontal: false // ✅ FIXED (no double flip)
        });

        const kp = pose.keypoints;

        // --- SQUAT LOGIC ---
        const hip = kp[11];
        const knee = kp[13];
        const ankle = kp[15];

        if (hip.score > 0.3 && knee.score > 0.3 && ankle.score > 0.3) {
            const angle = findAngle(
                hip.position,
                knee.position,
                ankle.position
            );

            ctx.fillStyle = "red";
            ctx.font = "20px Arial";
            ctx.fillText(Math.round(angle) + "°", knee.position.x, knee.position.y);

            if (angle < 110) sStage = "down";

            if (angle > 150 && sStage === "down") {
                sStage = "up";
                sCount++;
                document.getElementById("s-count").innerText = sCount;
            }
        }

        // --- HAND RAISE LOGIC ---
        const handsUp =
            (kp[9].position.y < kp[5].position.y) &&
            (kp[10].position.y < kp[6].position.y);

        if (handsUp && hStage === "down") {
            hStage = "up";
            hCount++;
            document.getElementById("h-count").innerText = hCount;
        } else if (!handsUp) {
            hStage = "down";
        }

        // --- DRAW ---
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // draw video
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        // draw skeleton + points
        drawSkeleton(kp);
        drawPoints(kp);

        requestAnimationFrame(detect);
    }

    detect();
}

main();