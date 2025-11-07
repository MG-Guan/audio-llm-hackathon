import { Component, OnInit } from '@angular/core';
import { ActivatedRoute } from '@angular/router';
import { Location } from '@angular/common';
import { CommonModule } from '@angular/common';
import { Submission } from '../../models/submission.model';
import { SubmissionService } from '../../services/submission.service';
import { ProgramService } from '../../services/program.service';
import { Program } from '../../models/program.model';
import { ProgressBarModule } from 'primeng/progressbar';
import { DividerModule } from 'primeng/divider';
import { ButtonModule } from 'primeng/button';

@Component({
  selector: 'app-submission-detail',
  standalone: true,
  imports: [CommonModule, ProgressBarModule, DividerModule, ButtonModule],
  styleUrls: ['./submission-detail.component.scss'],
  template: `
    <div class="submission-detail" *ngIf="submission">
      <!-- Overall Score and Rank -->
      <div class="score-section">
        <h2 class="text-lg font-bold text-gray-900">Overall Score</h2>
        <div class="score-value">{{ (submission.score.score * 100).toFixed(1) }}</div>
        <div class="rank" *ngIf="submission.rank">Rank: #{{ submission.rank }}</div>
      </div>

      <!-- Detailed Scores -->
      <div class="detailed-scores">       
        <!-- Audio Similarity Score -->
        <div class="score-item" *ngIf="submission.score.audioSimilarityScore !== undefined">
          <label>Audio Similarity</label>
          <p-progressBar 
            [value]="((submission.score.audioSimilarityScore || 0) * 100).toFixed(1)"
            [showValue]="true"
            [unit]="'%'"
          ></p-progressBar>
        </div>

        <!-- Text Similarity Score (if available) -->
        <div class="score-item" *ngIf="submission.score.textSimilarityScore !== undefined">
          <label>Text Similarity</label>
          <p-progressBar 
            [value]="((submission.score.textSimilarityScore || 0) * 100).toFixed(1)"
            [showValue]="true"
            [unit]="'%'"
          ></p-progressBar>
        </div>

        <div class="model-section" *ngIf="submission.score.model">
          <label class="font-bold text-gray-600">Scoring Model</label>
          <p class="text-sm text-gray-600">{{ submission.score.model }}</p>
        </div>
      </div>

      <p-divider></p-divider>

      <!-- Program Information -->
      <div class="program-section" *ngIf="program">
        <h2>Program Information</h2>
        <div>
          <h3>{{ program.title }}</h3>
          <p>{{ program.description }}</p>
        </div>
      </div>
    </div>

    <div class="try-again-section" style="width: 100%; text-align: center;">
      <button pButton (click)="location.back()" label="Try Again" class="try-button"></button>
    </div>
  `
})
export class SubmissionDetailComponent implements OnInit {
  submission?: Submission;
  program?: Program;
  rank?: number;

  constructor(
    private route: ActivatedRoute,
    private submissionService: SubmissionService,
    private programService: ProgramService,
    public location: Location
  ) {}

  ngOnInit() {
    this.route.params.subscribe(params => {
      this.submissionService.getSubmission(params['id']).subscribe(submission => {
        this.submission = submission;
        
        // Fetch program details
        if (submission?.programId) {
          this.programService.getProgram(submission.programId).subscribe(program => {
            this.program = program;
          });
        }
      });
    });
  }
}
