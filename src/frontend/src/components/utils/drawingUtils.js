// === DATA VALIDATION AND PROCESSING ===
const pixel_size = 4;
// Utility function to validate PSD data
export const validatePSDData = (psd) => {
  return Array.isArray(psd) && psd.length === 2048;
};

// Helper function to downsample PSD data
export const downsamplePsd = (psd, targetWidth) => {
  const step = Math.floor(psd.length / targetWidth);
  const downsampled = new Array(targetWidth);

  for (let i = 0; i < targetWidth; i++) {
    let sum = 0;
    const startIdx = i * step;
    const endIdx = Math.min(startIdx + step, psd.length);

    for (let j = startIdx; j < endIdx; j++) {
      sum += psd[j];
    }

    downsampled[i] = sum / (endIdx - startIdx);
  }

  return downsampled;
};

// Helper function to scale detection indices to match downsampled data
export const scaleDetections = (detections, originalLength, targetWidth) => {
  // If both detection values are 0, return [0, 0]
  if (detections[0] === 0 && detections[1] === 0) {
    return [0, 0];
  }

  const scale = targetWidth / originalLength;
  return [
    Math.floor(detections[0] * scale),
    Math.floor(detections[1] * scale)
  ];
};

// === IMAGE AND CANVAS OPERATIONS ===

const DetectionType = {
  NONE: 0,
  WIFI: 1,
  BLUETOOTH: 2,
};


export const createRowImageData = (
  width,
  psd,
  colorMap,
  detections,
  minDbm,
  maxDbm,
  detection,
  pixelSize = 8, // Default to 2 for backward compatibility
) => {
  // Downsample PSD data to match the canvas width divided by pixel size
  const pointsToDisplay = Math.floor(width / pixelSize);
  const downsampled = downsamplePsd(psd, pointsToDisplay);
  const scaledDetections = scaleDetections(detections, psd.length, pointsToDisplay);

  // Create a row with configurable pixel height
  const rowHeight = pixelSize;
  const imgData = new ImageData(width, rowHeight);
  const data = imgData.data;
  const dbmRange = maxDbm - minDbm;
  const [detStart, detEnd] = scaledDetections;

  for (let i = 0; i < pointsToDisplay; i++) {
    // Calculate normalized value more efficiently
    const normalized = Math.max(0, Math.min(1, (downsampled[i] - minDbm) / dbmRange));
    const colorIdx = Math.min(255, Math.floor(normalized * 255));
    let r, g, b, a;

    if (detection === DetectionType.WIFI && detStart < i && detEnd > i) {
      r = 0;
      g = 255;
      b = 0;
      a = 255; // pure blue
    } else if (detection === DetectionType.BLUETOOTH && detStart < i && detEnd > i) {
      r = 255;
      g = 0;
      b = 0;
      a = 255; // pure red
    } else {
      // Get color from the colormap
      [r, g, b, a] = colorMap[colorIdx] || [0, 0, 0, 255];
    }

    const alpha = (detStart < i && detEnd > i) ? a : a / 2;

    // Set all pixels in the NxN block
    for (let x = 0; x < pixelSize; x++) {
      for (let y = 0; y < rowHeight; y++) {
        const baseIdx = ((y * width) + (i * pixelSize) + x) * 4;
        data[baseIdx] = r;
        data[baseIdx + 1] = g;
        data[baseIdx + 2] = b;
        data[baseIdx + 3] = alpha;
      }
    }
  }

  return imgData;
};
export const shiftImageUp = (ctx, width, height, pixel_size) => {
  try {
    const image = ctx.getImageData(0, pixel_size, width, height - pixel_size);
    ctx.putImageData(image, 0, 0);
    return true;
  } catch (error) {
    console.error("Error shifting image data:", error);
    return false;
  }
};

export const clearCanvas = (ctx, width, height) => {
  ctx.clearRect(0, 0, width, height);
};

// === MAIN DRAWING FUNCTIONS ===

// Main drawing function
export const drawSpectrogram = ({
  canvas,
  psd,
  colorMap,
  detections,
  minDbm,
  maxDbm,
  detection,
  onError = () => { },
  drawTimeout = null,
  setDrawTimeout = () => { },
}) => {
  return new Promise((resolve, reject) => {
    try {
      // Validate inputs
      if (!canvas) {
        reject(new Error("Canvas not provided"));
        return;
      }

      if (!validatePSDData(psd)) {
        onError("Invalid PSD data format.");
        reject(new Error("Invalid PSD data format"));
        return;
      }

      // Check if previous draw is still in progress
      if (drawTimeout) {
        resolve();
        return;
      }

      // Setup canvas context
      const ctx = canvas.getContext("2d", { willReadFrequently: true });
      const width = canvas.width;
      const height = canvas.height;
      const pixel_size = 6;

      // Schedule drawing
      setDrawTimeout(
        requestAnimationFrame(() => {
          const imgData = createRowImageData(
            width,
            psd,             // Now passing the raw PSD data
            colorMap,
            detections,      // Now passing the raw detections
            minDbm,
            maxDbm,
            detection,
            pixel_size   // Specifying pixelSize as 2
          );

          // Shift existing image up
          if (!shiftImageUp(ctx, width, height, pixel_size)) {
            onError("Error shifting image data");
          }

          // Draw new row at the bottom
          ctx.putImageData(imgData, 0, height - pixel_size);

          // Clear timeout and resolve
          setDrawTimeout(null);
          resolve();
        }),
      );
    } catch (error) {
      console.error("Error in drawSpectrogram:", error);
      onError("Error drawing spectrogram.");
      reject(error);
    }
  });
};

export const createSpectrogramDrawer = (canvas, colorMap, onError) => {
  let currentDrawTimeout = null;
  let psdHistory = [];
  let detectionHistory = [];
  let detectionsHistory = [];
  let lastRangeDbm = [-110, -70]; // Default range
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  const maxHistorySize = canvas.height / pixel_size; // Each row is 2px high

  const redrawFullCanvas = (rangeDbm) => {
    clearCanvas(ctx, canvas.width, canvas.height);

    // Process each row of history from oldest to newest
    for (let i = 0; i < psdHistory.length; i++) {
      const psd = psdHistory[i];
      const detections = detectionsHistory[i];
      const detection = detectionHistory[i]
      // Calculate position from bottom (newest data at bottom)
      const y = canvas.height - ((psdHistory.length - i) * 2);
      if (y < 0) continue; // Skip if it would be drawn outside the canvas

      // Downsample and create row image data
      const pointsToDisplay = Math.floor(canvas.width / 2);
      const downsampled = downsamplePsd(psd, pointsToDisplay);
      const scaledDetections = scaleDetections(detections, psd.length, pointsToDisplay);
      const imgData = createRowImageData(
        canvas.width,
        downsampled,
        colorMap,
        scaledDetections,
        rangeDbm[0],
        rangeDbm[1],
        detection,
      );

      // Draw directly at the calculated position
      ctx.putImageData(imgData, 0, y);
    }
  };

  const draw = (psd, rangeDbm, detections, detection) => {
    // Update histories
    psdHistory.push(psd);
    detectionsHistory.push(detections);
    detectionHistory.push(detection);
    if (psdHistory.length > maxHistorySize) {
      detectionHistory.shift();
      psdHistory.shift();
      detectionHistory.shift();
    }

    lastRangeDbm = rangeDbm;

    return drawSpectrogram({
      canvas,
      psd,
      colorMap,
      detections,
      minDbm: rangeDbm[0],
      maxDbm: rangeDbm[1],
      detection,
      onError,
      drawTimeout: currentDrawTimeout,
      setDrawTimeout: (timeout) => {
        currentDrawTimeout = timeout;
      },
    });
  };

  const updateRange = (rangeDbm) => {
    // Cancel any pending draw
    if (currentDrawTimeout) {
      cancelAnimationFrame(currentDrawTimeout);
      currentDrawTimeout = null;
    }

    lastRangeDbm = rangeDbm;

    // Immediately redraw the entire canvas with the new range
    if (psdHistory.length > 0) {
      redrawFullCanvas(rangeDbm);
    }

    return Promise.resolve();
  };

  const clear = () => {
    clearCanvas(ctx, canvas.width, canvas.height);
    psdHistory = [];
    detectionHistory = [];
    detectionsHistory = [];
  };

  const cleanup = () => {
    if (currentDrawTimeout) {
      cancelAnimationFrame(currentDrawTimeout);
      currentDrawTimeout = null;
    }
    psdHistory = [];
    detectionHistory = [];
    detectionsHistory = [];
  };

  return {
    draw,
    updateRange,
    clear,
    cleanup,
  };
};
