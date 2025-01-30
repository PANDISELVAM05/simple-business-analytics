document.addEventListener("DOMContentLoaded", function () {
    const uploadForm = document.getElementById("upload-form");
    const forecastingImg = document.getElementById("forecasting-img");
    const comparisonImg = document.getElementById("comparison-img");
    const metricsTable = document.getElementById("metrics-table").getElementsByTagName("tbody")[0];
    const resultDiv = document.getElementById("result");

    uploadForm.addEventListener("submit", async function (event) {
        event.preventDefault(); // Prevent default form submission

        let formData = new FormData();
        let fileInput = document.querySelector("input[type='file']");

        if (fileInput.files.length === 0) {
            alert("Please select a CSV file to upload.");
            return;
        }

        formData.append("file", fileInput.files[0]);

        try {
            let response = await fetch("/upload", {
                method: "POST",
                body: formData
            });

            let result = await response.json();

            if (result.error) {
                alert("Error: " + result.error);
                return;
            }

            // Update forecasting and model comparison images
            forecastingImg.src = "/" + result.visualizations.forecasting_results;
            comparisonImg.src = "/" + result.visualizations.model_comparison;

            // Clear previous table data
            metricsTable.innerHTML = "";

            // Update Performance Metrics Table
            for (let model in result.results) {
                let row = metricsTable.insertRow();
                let cellModel = row.insertCell(0);
                let cellRMSE = row.insertCell(1);
                let cellMAE = row.insertCell(2);
                let cellR2 = row.insertCell(3);

                cellModel.textContent = model;
                cellRMSE.textContent = result.results[model].RMSE.toFixed(2);
                cellMAE.textContent = result.results[model].MAE.toFixed(2);
                cellR2.textContent = result.results[model].R2.toFixed(2);
            }

            // Function to add download links for results
            function addDownloadLinks() {
                let downloadDiv = document.createElement("div");
                downloadDiv.innerHTML = `
                    <h3>Download Results</h3>
                    <a href="C:/Users/Pandi selvam/Downloads/analysis_report.txt" download>📄 Download Report (TXT)</a><br>
                    <a href="C:/Users/Pandi selvam/Downloads/forecasting_results.png" download>📊 Download Forecasting Image</a><br>
                    <a href="C:/Users/Pandi selvam/Downloads/model_comparison.png" download>📈 Download Model Comparison Image</a><br>
                `;
                resultDiv.appendChild(downloadDiv);
            }

            // Call this function after successful upload
            setTimeout(addDownloadLinks, 2000);

        } catch (error) {
            console.error("Error:", error);
            alert("An error occurred while uploading the file.");
        }
    });
});
