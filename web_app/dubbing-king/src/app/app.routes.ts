import { Routes } from '@angular/router';
import { HomeComponent } from './components/home/home.component';
import { ProgramsComponent } from './components/programs/programs.component';
import { ProgramDetailComponent } from './components/program-detail/program-detail.component';
import { LeaderboardComponent } from './components/leaderboard/leaderboard.component';
import { RecordSubmissionComponent } from './components/record-submission/record-submission.component';
import { SubmissionDetailComponent } from './components/submission-detail/submission-detail.component';

export const routes: Routes = [
  { path: '', component: HomeComponent },
  { path: 'programs', component: ProgramsComponent },
  { path: 'program/:id', component: ProgramDetailComponent },
  { path: 'program/leaderboard/:id', component: LeaderboardComponent },
  { path: 'program/record/:id', component: RecordSubmissionComponent },
  { path: 'submission/:id', component: SubmissionDetailComponent },
  { path: '**', redirectTo: '' }
];