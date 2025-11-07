export enum RecordingState {
  IDLE = 'IDLE',
  RECORDING = 'RECORDING',
  RECORDED = 'RECORDED',
  PLAYING = 'PLAYING',
  SUBMITTING = 'SUBMITTING'
}

export interface RecordingStatus {
  state: RecordingState;
  time: string;
  audio?: Blob;
}
