import { SubmissionScore } from './submission-score.model';

export interface Submission {
    id: string;
    programId: string;
    clipId: string;
    username: string;
    audioUrl: string;
    score: SubmissionScore;
    rank?: number;
}

export interface PostSubmissionRequest {
    programId: string;
    clipId: string;
    username: string;
    audio: Blob;
    model?: string;
    token: string;
}

export interface SubmissionAPIResponse {
    id: string;
    program_id: string;
    clip_id: string;
    username: string;
    file_id: string;
    score: number;
    model: string;
    audio_similarity_score?: number;
    text_similarity_score?: number;
    rank?: number;
    total_submissions?: number;
}