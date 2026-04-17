export type Point = [number, number];
export type Box = Point[];

export interface OcrLine {
  text: string;
  conf: number;
  box: Box;
}

export interface OcrDebug {
  detDims: number[];
  detShape: number[];
  probMin: number;
  probMax: number;
  probMean: number;
  pixelsAboveThresh: number;
  rawComponents: number;
  survivedMinSize: number;
  survivedScore: number;
  survivedFinalSize: number;
  scoreMaxOfRejected: number;
  boxesDetected: number;
  linesRecognized: number;
  dictSize: number;
  recOutputDims: number[] | null;
  recVocabSize: number | null;
  firstBoxArgmax: number[] | null;
  firstBoxMaxIndex: number | null;
  firstBoxWH: [number, number] | null; // [width, height] pierwszego boxa w px oryginalnego obrazu
}

export interface OcrResponse {
  n_lines: number;
  mean_conf: number;
  time_s: number;
  lines: OcrLine[];
  debug?: OcrDebug;
}
