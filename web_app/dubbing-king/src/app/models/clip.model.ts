export interface Clip {
    id: string;
    duration: number;
    transcript: string;
    speaker?: string;
    audioUrl?: string;
}