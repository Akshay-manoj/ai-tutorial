export function chunkText(text, { chunkSize = 500, overlap = 50 }) {
  if (chunkSize <= 0) {
    throw new Error("chunkSize must be greater than 0.");
  }
  if (overlap < 0 || overlap >= chunkSize) {
    throw new Error("overlap must be between 0 and chunkSize - 1.");
  }

  const chunks = [];
  let start = 0;

  while (start < text.length) {
    chunks.push(text.slice(start, start + chunkSize).trim());

    if (start + chunkSize >= text.length) break;
    start += chunkSize - overlap;
  }

  return chunks.filter(Boolean);
}