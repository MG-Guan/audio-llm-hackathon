import { Component, OnInit } from '@angular/core';
import { ActivatedRoute } from '@angular/router';
import { CommonModule } from '@angular/common';
import { Program } from '../../models/program.model';
import { ProgramService } from '../../services/program.service';
import { switchMap } from 'rxjs/operators';
import { TagModule } from 'primeng/tag';
import { DividerModule } from 'primeng/divider';
import { ButtonModule } from 'primeng/button';
import { RouterLink } from '@angular/router';
import { TabsModule } from 'primeng/tabs';
import { TableModule } from 'primeng/table';
import { SubmissionService } from '../../services/submission.service';
import { ModelService } from '../../services/model.service';
import { Submission } from '../../models/submission.model';
import { SpeakerSeverityService } from '../../services/speaker-severity.service';

@Component({
  selector: 'app-program-detail',
  standalone: true,
  imports: [CommonModule, TagModule, DividerModule, ButtonModule, RouterLink, TabsModule, TableModule],
  styleUrls: ['./program-detail.component.scss'],
  template: `
    <div class="program-detail" *ngIf="program">
      <div class="video-container">
        <!-- Video player placeholder - to be implemented with actual video player -->
        <div class="video-player">
          <video playsinline [src]="program.videoUrl" controls></video>
        </div>
      </div>
      
      <div class="content">
        <h1 class="title">{{program.title}}</h1>
        <p class="description">{{program.description}}</p>
      </div>

      <div class="tabs-container">
        <p-tabs value="0">
          <p-tablist>
            <p-tab value="0">Leaderboard</p-tab>
            <p-tab value="1">Transcripts</p-tab>
          </p-tablist>
          <p-tabpanels>
            <p-tabpanel value="0">
              <div class="card">
                <p class="text-secondary text-sm text-gray-900" style="text-align: center; margin-bottom: 0.5rem;">You are now viewing the leaderboard for the model: {{model}}</p>
                <p-table [value]="submissions" [tableStyle]="{ width: '100%' }">
                  <ng-template pTemplate="header">
                    <tr>
                      <th style="width: 30px">Rank</th>
                      <th style="width: 60px">Score</th>
                      <th>Submitter</th>
                      <th style="width: 30px">Listen</th>
                    </tr>
                  </ng-template>
                  <ng-template pTemplate="body" let-submission let-index="rowIndex">
                    <tr>
                      <td class="text-center font-bold">
                        {{index + 1}}
                      </td>
                      <td class="text-right score-container">
                        <p class="text-lg font-bold">{{(submission.score.score * 100).toFixed(1)}}</p>
                        <div class="detail-score">
                          <i *ngIf="submission.score.audioSimilarityScore" class="pi pi-volume-up">{{submission.score.audioSimilarityScore?.toFixed(1)}}</i>
                          <i *ngIf="submission.score.textSimilarityScore" class="pi pi-align-left">{{submission.score.textSimilarityScore?.toFixed(1)}}</i>
                        </div>
                      </td>
                      <td>{{submission.username}}</td>
                      <td class="text-center">
                        <button 
                          pButton 
                          icon="pi pi-play" 
                          class="p-button-rounded p-button-text"
                          (click)="playAudio(submission.audioUrl)"
                        ></button>
                      </td>
                    </tr>
                  </ng-template>
                </p-table>
              </div>
            </p-tabpanel>
            <p-tabpanel value="1">
              <div class="card">
                <div class="transcript-list">
                  <div *ngFor="let clip of program.clips" class="transcript-item">
                    <div class="transcript-info">
                      <p-tag value="{{clip.speaker || 'Speaker'}}" [severity]="getSeverityForSpeaker(clip.speaker)"></p-tag>
                      <div class="clip-duration">{{formatDuration(clip.duration)}}</div>
                    </div>
                    <div class="transcript-text-container">
                      <p class="transcript-text text-sm">{{clip.transcript}}</p>
                    </div>
                  </div>
                </div>
              </div>
            </p-tabpanel>
          </p-tabpanels>
        </p-tabs>
      </div>

      <div class="action-bar">
        <!-- <button pButton label="Leaderboard" icon="pi pi-align-justify" [routerLink]="['/program', 'leaderboard', program.id]"></button> -->
        <button pButton label="Start Dubbing" icon="pi pi-play" [routerLink]="['/program', 'record', program.id]"></button>
      </div>
    </div>
  `
})
export class ProgramDetailComponent implements OnInit {
  program?: Program;
  submissions: Submission[] = [];
  model: string = '';
  private audioPlayer?: HTMLAudioElement;

  constructor(
    private route: ActivatedRoute,
    private programService: ProgramService,
    private submissionService: SubmissionService,
    private modelService: ModelService,
    private speakerSeverityService: SpeakerSeverityService
  ) {
    this.model = this.modelService.getModel();
  }

  ngOnInit() {
    this.route.params.pipe(
      switchMap(params => this.programService.getProgram(params['id']))
    ).subscribe(program => {
      this.program = program;
    });

    // Get submissions for leaderboard
    this.route.params.pipe(
      switchMap(params => this.submissionService.getLeaderboard(params['id'], this.model))
    ).subscribe(submissions => {
      this.submissions = submissions;
    });

    // Initialize audio player
    this.audioPlayer = document.createElement('audio');
    this.audioPlayer.style.display = 'none';
    document.body.appendChild(this.audioPlayer);
  }

  playAudio(audioUrl: string) {
    if (this.audioPlayer) {
      this.audioPlayer.src = audioUrl;
      this.audioPlayer.play();
    }
  }

  ngOnDestroy() {
    // Cleanup audio player
    if (this.audioPlayer) {
      document.body.removeChild(this.audioPlayer);
    }
  }

  formatDuration(seconds: number): string {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
  }

  getSeverityForSpeaker(speaker: string | undefined): 'secondary' | 'success' | 'info' | 'warn' | 'danger' {
    return this.speakerSeverityService.getSeverityForSpeaker(speaker);
  }
}
