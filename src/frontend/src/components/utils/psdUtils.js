export const downsamplePsd = (psd, targetWidth) => {
  const step = Math.floor(psd.length / targetWidth);
  const downsampled = [];
  for (let i = 0; i < targetWidth; i++) {
    const segment = psd.slice(i * step, (i + 1) * step);
    const avg = segment.reduce((a, b) => a + b, 0) / step;
    downsampled.push(avg);
  }
  return downsampled;
};

export const generateStructuredPsd = (timeIndex) => {
  const psd = new Array(2048).fill(0);

  // Baseline noise
  for (let i = 0; i < 2048; i++) {
    psd[i] = 0.2 + Math.random() * 0.1;
  }

  // Stable horizontal lines
  const stableLines = [100, 200, 400, 800, 1200, 1600];
  stableLines.forEach((freqIndex) => {
    psd[freqIndex] += 0.6 + Math.random() * 0.2;
  });

  // Vertical band in the center
  for (let i = 900; i <= 1140; i++) {
    psd[i] += 0.3;
  }

  // Random flicker
  if (timeIndex % 15 === 0) {
    for (let i = 600; i <= 650; i++) {
      psd[i] += Math.random() * 0.3;
    }
  }

  // Clamp values
  return psd.map((val) => Math.min(1, val));
};
