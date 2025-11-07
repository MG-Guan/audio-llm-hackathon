import { Component } from '@angular/core';
import { RouterLink } from '@angular/router';
import { ButtonModule } from 'primeng/button';

@Component({
  selector: 'app-home',
  standalone: true,
  imports: [RouterLink, ButtonModule],
  template: `
    <div class="home-container">
      <div class="content-wrapper">
        <h1 class="title">Dubbing King</h1>
        <p class="subtitle">
          Try your best dubbing for movie clips and see how you score on similarity!
        </p>
        <button pButton routerLink="/programs" label="Try Now" class="try-button"></button>
      </div>
    </div>
  `,
  styles: [`
    .home-container {
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 1rem;
    }
    
    .content-wrapper {
      text-align: center;
      color: white;
    }
    
    .title {
      font-size: 4rem;
      font-weight: bold;
      margin-bottom: 1rem;
      text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    .subtitle {
      font-size: 1.5rem;
      margin-bottom: 2rem;
      max-width: 600px;
      line-height: 1.4;
    }
    
    .try-button {
      font-size: 1.25rem;
      padding: 1rem 2.5rem;
      border-radius: 2rem;
      background: white;
      color: #9c27b0;
      border: none;
      transition: all 0.3s ease;
      
      &:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
      }
    }
  `]
})
export class HomeComponent {}