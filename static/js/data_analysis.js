(function () {
  "use strict";

  /* Module state */
  let initialized = false;
  let elements = null;
  let helpers = null;

  /* Initialize the Data Analysis module */
  function initialize(options) {
    if (initialized) {
      return;
    }

    elements = options?.elements;
    helpers = options?.helpers;

    if (!elements?.dataUpload) {
      console.error("Missing Data Analysis upload button." );
      return;
    }

    const requiredHelpers = [
      "getCurrentMode",
      "isProcessing",
      "addBubble",
      "setProcessingState",
    ];

    const missingHelper = requiredHelpers.find(
      (name) =>
        typeof helpers?.[name] !== "function"
    );

    if (missingHelper) {
      console.error(`Missing Data Analysis helper: ${missingHelper}`);
      return;
    }

    elements.dataUpload.addEventListener("click",uploadDataset);
    initialized = true;
  }

  /* Upload a CSV or Excel dataset to the backend */
  async function uploadDataset() {
    if (helpers.getCurrentMode() !== "data") {
      return;
    }

    if (helpers.isProcessing()) {
      return;
    }

    const file = elements.dataFile?.files?.[0];

    if (!file) {
      helpers.addBubble(
        "assistant",
        "Please choose a CSV or Excel file first."
      );

      return;
    }

    if (!isSupportedFile(file.name)) {
      helpers.addBubble(
        "assistant",
        "Only CSV, XLSX, and XLS files are supported."
      );

      return;
    }

    helpers.setProcessingState(true);
    updateFileStatus("Loading dataset…");

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("/api/data/upload", {
          method: "POST",
          body: formData,
        }
      );

      const result = await readJsonResponse(response);

      if (!response.ok || !result.ok) {
        throw new Error(
          result.error || `Upload failed with HTTP ${response.status}.`
        );
      }

      updateFileStatus(
        `${result.filename} · ${result.rows} rows · ${result.columns} columns`
      );

      if (elements.chatLog) {
        elements.chatLog.innerHTML = "";
      }
      helpers.addBubble(
        "assistant",
        result.reply || "Dataset loaded successfully."
      );

    } catch (error) {
      updateFileStatus("Dataset loading failed");

      helpers.addBubble(
        "assistant",
        `Dataset upload failed: ${error.message}`
      );
    } finally {
      helpers.setProcessingState(false);
    }
  }

   /* Check whether the selected file extension is supported */
  function isSupportedFile(filename) {
    const normalizedFilename = String(filename || "").toLowerCase();

    return [".csv", ".xlsx", ".xls"].some((extension) =>
        normalizedFilename.endsWith(extension)
    );
  }

  /* Read the JSON response returned by the server */
  async function readJsonResponse(response) {
    try {
      return await response.json();
    } catch {
      throw new Error(
        `The server returned HTTP ${response.status} without a valid JSON response.`
      );
    }
  }

  /* Update the dataset status displayed in the interface */
  function updateFileStatus(message) {
    if (elements.dataFileStatus) {
      elements.dataFileStatus.textContent = message;
    }
  }

  /* Clear the selected dataset and reset the status label */
  function clearDatasetUI() {
    if (elements.dataFile) {
      elements.dataFile.value = "";
    }

    updateFileStatus("No dataset loaded");
  }

   /* Show controls specific to Data Analysis mode */
  function showModeUI() {
    document.body.classList.add("data-mode");

    if (elements.settingsRow) {
      elements.settingsRow.classList.remove("hidden");
    }

    if (elements.dataControls) {
      elements.dataControls.classList.remove("hidden");
    }

    if (elements.modeSubtitle) {
      elements.modeSubtitle.textContent =
        "Data Analysis · Analyze CSV and Excel datasets";
    }
  }

  /* Hide controls when leaving Data Analysis mode */
  function hideModeUI() {
    document.body.classList.remove("data-mode");

    if (elements.dataControls) {
      elements.dataControls.classList.add("hidden");
    }
  }

  /* Return the predefined suggestion prompts */
  function getSuggestions() {
    return [
      {
        label: "Dataset Summary",
        value: "Summarize this dataset.",
      },
      {
        label: "Missing Values",
        value: "Which columns contain missing values?",
      },
      {
        label: "Numeric Statistics",
        value:
          "Explain the numeric descriptive statistics.",
      },
      {
        label: "Column Types",
        value:
          "Describe the columns and their data types.",
      },
      {
        label: "Patterns",
        value:
          "What useful patterns are visible in the dataset summary?",
      },
      {
        label: "Data Quality",
        value:
          "Identify possible data-quality issues.",
      },
    ];
  }

  /* Build the request payload for the backend */
  function getRequest(text) {
    return {
      endpoint: "/api/data",
      payload: {
        message: text,
      },
    };
  }

  window.DataAnalysisMode = {
    initialize,
    showModeUI,
    hideModeUI,
    clearDatasetUI,
    getSuggestions,
    getRequest,
  };
})();