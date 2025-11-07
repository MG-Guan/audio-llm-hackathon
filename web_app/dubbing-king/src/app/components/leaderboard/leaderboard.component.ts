import { Component, OnInit } from '@angular/core';
import { ActivatedRoute } from '@angular/router';
import { CommonModule } from '@angular/common';
import { TableModule } from 'primeng/table';
import { ButtonModule } from 'primeng/button';
import { SubmissionService } from '../../services/submission.service';
import { Submission } from '../../models/submission.model';
import { switchMap } from 'rxjs/operators';
import { ProgramCardComponent } from '../program-card/program-card.component';
import { ProgramService } from '../../services/program.service';
import { Program } from '../../models/program.model';
import { ModelService } from '../../services/model.service';

@Component({
  selector: 'app-leaderboard',
  standalone: true,
  imports: [CommonModule, TableModule, ButtonModule, ProgramCardComponent],
  styleUrls: ['./leaderboard.component.scss'],
  template: `
    <div class="leaderboard-container">
      <div class="program-info">
        <app-program-card *ngIf="program" [program]="program"></app-program-card>
      </div>
      <h1 class="text-2xl font-bold text-white mb-4" style="text-align: center; margin-bottom: 0.5rem;">Leaderboard</h1>
      <p class="text-secondary text-sm text-white" style="text-align: center; margin-bottom: 0.5rem;">You are now viewing the leaderboard for the model: {{model}}</p>
      
      <div class="card">
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

      <audio #audioPlayer style="display: none"></audio>
    </div>
  `
})
export class LeaderboardComponent implements OnInit {
  submissions: Submission[] = [];
  program?: Program;
  model: string = '';
  private audioPlayer?: HTMLAudioElement;
  
  constructor(
    private route: ActivatedRoute,
    private submissionService: SubmissionService,
    private programService: ProgramService,
    private modelService: ModelService
  ) {
    this.model = this.modelService.getModel();
  }

  ngOnInit() {
    // Get program
    this.route.params.pipe(
      switchMap(params => this.programService.getProgram(params['id']))
    ).subscribe(program => {
      this.program = program;
    });

    // Get submissions
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
}
