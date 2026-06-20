const predbtn = document.getElementById("predbtn");
const ImageInput = document.getElementById("ImageInput");
const result = document.getElementById("result");
const preview = document.getElementById("preview");
const animal = document.getElementById("animal");
const confidenceText = document.getElementById("confidence");
const progressBar = document.getElementById("progress-bar");
const filename = document.getElementById("filename");
const shapImg = document.getElementById("shap-img")


predbtn.addEventListener("click", async (event) => {
    event.preventDefault();
    console.log("BUTTON CLICKED");

    const file = ImageInput.files[0];

    if (!file) {
        alert("Select an image first");
        return;
    }

    try {

        const formdata = new FormData();
        formdata.append("file", file);

        console.log("SENDING REQUEST");

        const response = await fetch(
            "http://localhost:8000/predict",
            {
                method: "POST",
                body: formdata
            }
        );

        console.log("RESPONSE RECEIVED", response.status, response.ok);

        const text = await response.text();
        console.log("RAW RESPONSE:", text);

        if (!response.ok) {
            throw new Error(`API error: ${response.status} - ${text}`);
        }

        const data = JSON.parse(text);

        console.log("DATA:", data);

        animal.innerText = data.Verdict;

        confidenceText.innerText =
            `Confidence: ${(data.Prediction * 100).toFixed(2)}%`;

        progressBar.style.width =
            `${(data.Prediction * 100).toFixed(2)}%`;
        shapImg.src =
            `http://localhost:8000/static/${data.shap_image}`;

        shapImg.style.display = "block";
    }
    catch (err) {

        console.error(err);

    }

});

ImageInput.addEventListener("change", () => {

    const file = ImageInput.files[0];

    if (file) {

        filename.innerText = file.name;

        preview.src = URL.createObjectURL(file);

        preview.style.display = "block";
    }

});
const tabs =
    document.querySelectorAll(".tab-btn");

const contents =
    document.querySelectorAll(".tab-content");

tabs.forEach(btn => {

    btn.addEventListener("click", () => {

        contents.forEach(content => {

            content.classList.remove("active-tab");

        });

        tabs.forEach(tab => {

            tab.classList.remove("active");

        });

        btn.classList.add("active");

        document
            .getElementById(btn.dataset.tab)
            .classList.add("active-tab");

    });

});