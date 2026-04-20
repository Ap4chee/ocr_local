export type Point = [number, number];
export type Box = Point[];

export interface OcrLine {
  text: string;
  conf: number;
  box: Box;
}

export interface OcrResponse {
  n_lines: number;
  mean_conf: number;
  time_s: number;
  lines: OcrLine[];
}
