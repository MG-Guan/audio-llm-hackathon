export interface SubmissionScore {
    model: string;
    score: number;
    audioSimilarityScore?: number;
    textSimilarityScore?: number;
}