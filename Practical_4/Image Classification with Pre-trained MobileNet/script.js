async function compareModels() {

    console.clear();
    console.log("===== MODEL COMPARISON STARTED =====");

    const resultsDiv = document.getElementById("results");
    resultsDiv.innerHTML = "Loading models... Please wait.<br><br>";

    // Load Models
    const mobileNetModel = await mobilenet.load();
    const cocoModel = await cocoSsd.load();

    console.log("Models Loaded Successfully");

    resultsDiv.innerHTML = "";

    for (let i = 1; i <= 5; i++) {

        const img = document.getElementById("img" + i);

        const mobilePred = await mobileNetModel.classify(img);
        const cocoPred = await cocoModel.detect(img);

        console.log("--------------------------------------------------");
        console.log("Image " + i);

        let output = "<b>Image " + i + "</b><br><br>";

        // MobileNet Predictions
        output += "<b>MobileNet (Classification):</b><br>";
        console.log("MobileNet Predictions:");

        let mobileTable = [];

        for (let j = 0; j < 3; j++) {

            const label = mobilePred[j].className;
            const prob = (mobilePred[j].probability * 100).toFixed(2);

            mobileTable.push({ Label: label, Confidence: prob + "%" });

            output += label + " - " + prob + "%<br>";
        }

        console.table(mobileTable);

        // COCO-SSD Predictions
        output += "<br><b>COCO-SSD (Object Detection):</b><br>";
        console.log("COCO-SSD Detection:");

        if (cocoPred.length > 0) {

            let cocoTable = [];

            cocoPred.forEach(p => {

                const label = p.class;
                const score = (p.score * 100).toFixed(2);

                cocoTable.push({ Object: label, Confidence: score + "%" });

                output += label + " - " + score + "%<br>";
            });

            console.table(cocoTable);

        } else {

            console.log("No objects detected");
            output += "No objects detected<br>";
        }

        output += "<hr>";
        resultsDiv.innerHTML += output;
    }

    console.log("===== MODEL COMPARISON FINISHED =====");
}