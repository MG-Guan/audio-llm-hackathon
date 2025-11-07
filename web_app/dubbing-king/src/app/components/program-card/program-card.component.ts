import { Component, Input } from '@angular/core';
import { RouterLink } from '@angular/router';
import { TagModule } from 'primeng/tag';
import { Program } from '../../models/program.model';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-program-card',
  standalone: true,
  imports: [RouterLink, TagModule, CommonModule],
  template: `
    <div class="program-card" [routerLink]="['/program', program.id]">
      <div class="thumbnail">
        <img [src]="program.thumbnailUrl" [alt]="program.title">
        <span class="duration">{{formatDuration(program.duration)}}</span>
      </div>
      <div class="content">
        <h2 class="text-lg font-bold text-gray-900">{{program.title}}</h2>
        <p class="description text-sm text-gray-500">{{program.description}}</p>
        <div class="meta">
          <div class="tags">
            <p-tag *ngFor="let tag of program.tags" [value]="tag" severity="info"></p-tag>
          </div>
        </div>
      </div>
    </div>
  `,
  styles: [`
    .program-card {
      display: flex;
      background: rgba(255, 255, 255, 0.9);
      border-radius: 12px;
      overflow: hidden;
      margin-bottom: 1rem;
      cursor: pointer;
      transition: transform 0.2s, box-shadow 0.2s;

      &:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
      }
    }

    .thumbnail {
      position: relative;
      top: 0;
      left: 0;
      bottom: 0;
      width: 120px;

      img {
        width: 100%;
        height: 100%;
        object-fit: cover;
      }

      .duration {
        position: absolute;
        bottom: 8px;
        right: 8px;
        background: rgba(0, 0, 0, 0.7);
        color: white;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 0.8rem;
      }
    }

    .content {
      flex: 1;
      padding: 1rem;

      h2 {
        margin: 0 0 0.5rem;
        color: var(--primary-color);
      }

      .description {
        color: var(--text-color);
        margin-bottom: 1rem;
      }
    }

    .meta {
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    .tags {
      display: flex;
      gap: 0.5rem;

      p-tag {
        font-size: 0.8rem;
      }
    }
  `]
})
export class ProgramCardComponent {
  @Input() program!: Program;

  formatDuration(seconds: number): string {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
  }
}
