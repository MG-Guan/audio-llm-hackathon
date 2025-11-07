import { Component } from '@angular/core';

@Component({
  selector: 'app-animated-background',
  standalone: true,
  template: `
    <div class="animated-gradient"></div>
  `,
  styles: [`
    .animated-gradient {
      position: fixed;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      background: linear-gradient(135deg, #9c27b0, #2196f3);
      background-size: 400% 400%;
      animation: gradient 15s ease infinite;
      z-index: -1;
    }

    @keyframes gradient {
      0% {
        background-position: 0% 50%;
      }
      50% {
        background-position: 100% 50%;
      }
      100% {
        background-position: 0% 50%;
      }
    }
  `]
})
export class AnimatedBackgroundComponent {}
