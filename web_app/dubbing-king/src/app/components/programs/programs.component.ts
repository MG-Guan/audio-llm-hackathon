import { Component, OnInit } from '@angular/core';
import { Program } from '../../models/program.model';
import { ProgramService } from '../../services/program.service';
import { ProgramCardComponent } from '../program-card/program-card.component';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-programs',
  standalone: true,
  imports: [ProgramCardComponent, CommonModule],
  template: `
    <div class="programs-container">
      <h1 class="text-2xl font-bold text-white mb-4">Choose a video to dub!</h1>
      <div class="program-list">
        <app-program-card 
          *ngFor="let program of programs" 
          [program]="program"
        ></app-program-card>
      </div>
    </div>
  `,
  styles: [`
    .programs-container {
      padding: 1.2rem;
      max-width: 1200px;
      margin: 60px auto 0;
      
      h1 {
        color: white;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
      }
    }
  `]
})
export class ProgramsComponent implements OnInit {
  programs: Program[] = [];

  constructor(private programService: ProgramService) {}

  ngOnInit() {
    this.programService.getPrograms().subscribe(programs => {
      this.programs = programs;
    });
  }
}