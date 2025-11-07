import { inject, Injectable } from '@angular/core';
import { map, Observable, of } from 'rxjs';
import { Submission, PostSubmissionRequest, SubmissionAPIResponse } from '../models/submission.model';
import config from './config';
import { HttpClient } from '@angular/common/http';

const { apiUrl } = config;

@Injectable({
    providedIn: 'root'
})
export class SubmissionService {
    private http = inject(HttpClient);

    private apiSubmissionToSubmission = (submission: SubmissionAPIResponse): Submission => {
        return {
            id: submission.id,
            programId: `${submission.program_id}[${submission.clip_id}]`,
            clipId: '1',
            username: submission.username,
            audioUrl: `${apiUrl}/files/${submission.file_id}`,
            rank: submission.rank,
            score: {
                score: submission.score,
                model: submission.model,
                audioSimilarityScore: submission.audio_similarity_score,
                textSimilarityScore: submission.text_similarity_score
            }
        };
    };

    private appProgramIdToServerId = (programId: string): { program_id: string, clip_id: string } => {
      const match = programId.match(/\[([^\]]+)\](?!.*\[)$/);
      const postProgramId = programId.slice(0, -(match?.[0]?.length ?? 0));
      const postClipId = match?.[1];
      if (!postProgramId || !postClipId) {
        throw new Error('Failed to get program and clip IDs');
      }
      return {
        program_id: postProgramId,
        clip_id: postClipId
      };
    }

    private serverIdToAppProgramId = (programId: string, clipId: string): string => {
        return `${programId}[${clipId}]`;
    }

    getSubmissions(programId: string): Observable<Submission[]> {
        const { program_id: serverProgramId, clip_id: serverClipId } = this.appProgramIdToServerId(programId);
        return this.http.get<Submission[]>(`${apiUrl}/submissions?program_id=${serverProgramId}`);
    }

    getSubmission(id: string): Observable<Submission | undefined> {
        return this.http.get<SubmissionAPIResponse>(`${apiUrl}/submissions/${id}`).pipe(
            map(this.apiSubmissionToSubmission)
        );
    }

    getSubmissionsOrderByScore(programId: string, model?: string, direction: 'asc' | 'desc' = 'desc', limit: number = 10, offset: number = 0): Observable<Submission[]> {
        const { program_id: serverProgramId, clip_id: serverClipId } = this.appProgramIdToServerId(programId);
        return this.http.get<SubmissionAPIResponse[]>(`${apiUrl}/submissions?program_id=${serverProgramId}&clip_id=${serverClipId}&sort_by=score&sort_direction=${direction}&limit=${limit}&offset=${offset}&model=${model}`).pipe(
            map(submissions => submissions.map(this.apiSubmissionToSubmission))
        );
    }

    getLeaderboard(programId: string, model?: string, limit: number = 20): Observable<Submission[]> {
        const { program_id: serverProgramId, clip_id: serverClipId } = this.appProgramIdToServerId(programId);
        return this.http.get<SubmissionAPIResponse[]>(`${apiUrl}/leaderboard?program_id=${serverProgramId}&clip_id=${serverClipId}&limit=${limit}&model=${model}`).pipe(
            map(submissions => submissions.map(this.apiSubmissionToSubmission))
        );
    }

    async postSubmission(postRequest: PostSubmissionRequest): Promise<Submission> {
        // Step 1: Upload the audio file
        const formData = new FormData();
        formData.append('file', postRequest.audio);
        formData.append('token', postRequest.token);

        const uploadResponse = await fetch(`${config.apiUrl}/upload`, {
            method: 'POST',
            body: formData
        });

        if (!uploadResponse.ok) {
            throw new Error('Failed to upload audio file');
        }

        const { file_id } = await uploadResponse.json();

        // Step 2: Create submission with the file_id
        const submissionResponse = await fetch(`${config.apiUrl}/submissions`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                username: postRequest.username,
                file_id: file_id,
                model: postRequest.model,
                ...this.appProgramIdToServerId(postRequest.programId)
            })
        });

        if (!submissionResponse.ok) {
            throw new Error('Failed to create submission');
        }

        const submissionData = await submissionResponse.json();
        
        // Transform the response to match our frontend model
        return {
            id: submissionData.id,
            programId: submissionData.program_id,
            clipId: submissionData.clip_id,
            username: submissionData.username,
            audioUrl: submissionData.audio_url,
            score: {
                model: submissionData.score.model,
                score: submissionData.score.score,
                audioSimilarityScore: submissionData.score.audio_similarity_score,
                textSimilarityScore: submissionData.score.text_similarity_score
            }
        };
    }
}