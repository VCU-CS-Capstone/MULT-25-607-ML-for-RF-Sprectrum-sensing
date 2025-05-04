// psdUtils.js

/**
 * Validate PSD data array.
 * @param {Array} psd
 * @returns {boolean}
 */
export function isValidPSD(psd) {
  return Array.isArray(psd) && psd.length === 2048;
}

/**
 * Downsample PSD data to a target width.
 * @param {Array<number>} psd
 * @param {number} targetWidth
 * @returns {Array<number>}
 */
export function downsamplePSD(psd, targetWidth) {
  const step = Math.floor(psd.length / targetWidth);
  const downsampled = [];
  for (let i = 0; i < targetWidth; i++) {
    const segment = psd.slice(i * step, (i + 1) * step);
    const avg = segment.reduce((a, b) => a + b, 0) / step;
    downsampled.push(avg);
  }
  return downsampled;
}

/**
 * Scale detection indices to match downsampled data.
 * @param {[number, number]} detections
 * @param {number} originalLength
 * @param {number} targetWidth
 * @returns {[number, number]}
 */
export function scaleDetections(detections, originalLength, targetWidth) {
  if (detections[0] === 0 && detections[1] === 0) return [0, 0];
  const scale = targetWidth / originalLength;
  return [Math.floor(detections[0] * scale), Math.floor(detections[1] * scale)];
}
