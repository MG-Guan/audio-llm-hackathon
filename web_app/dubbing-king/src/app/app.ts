import { Component } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { MenubarModule } from 'primeng/menubar';
import { DialogModule } from 'primeng/dialog';
import { ButtonModule } from 'primeng/button';
import { InputTextModule } from 'primeng/inputtext';
import { FormsModule } from '@angular/forms';
import { AvatarModule } from 'primeng/avatar';
import { AnimatedBackgroundComponent } from './components/animated-background/animated-background.component';
import { CommonModule } from '@angular/common';
import { UserService } from './services/user.service';
import { ModelService } from './services/model.service';
import { MenuItem } from 'primeng/api';
import { DividerModule } from 'primeng/divider';
import { SelectModule } from 'primeng/select';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [
    RouterOutlet, 
    MenubarModule, 
    DialogModule,
    ButtonModule,
    InputTextModule,
    FormsModule,
    AvatarModule,
    CommonModule,
    AnimatedBackgroundComponent,
    DividerModule,
    SelectModule,
  ],
  styleUrls: ['./app.scss'],
  template: `
    <div class="app-container">
      <app-animated-background></app-animated-background>
      <p-menubar class="user-menu" [model]="items">
        <ng-template pTemplate="end">
          <div class="user-menu">
            <span class="username-text">{{ username }}</span>
            <p-avatar
              icon="pi pi-user"
              shape="circle"
              size="large"
              [style]="{ cursor: 'pointer', backgroundColor: 'var(--primary-color)' }"
              (click)="showDialog()"
            ></p-avatar>
            <i class="pi pi-github" style="font-size: 1.5rem; color: var(--text-color); cursor: pointer;" (click)="navigateToGitHub()"></i>
          </div>
        </ng-template>
      </p-menubar>
      <p-dialog 
        header="Configuration"
        [(visible)]="visible" 
        [modal]="true"
        [draggable]="false"
        [resizable]="false"
      >
        <div class="flex flex-col gap-4">
          <div class="flex items-center gap-4">
            <label for="username" class="font-semibold w-24">Username</label>
            <input 
              pInputText 
              id="username" 
              class="flex-auto" 
              [(ngModel)]="username"
              placeholder="Enter username"
              autocomplete="off" 
            />
          </div>
          <span class="text-secondary">Enter your preferred username below. <br />Will be used to identify you in the leaderboard.</span>
          <p-divider />
          <div class="flex items-center gap-4">
            <label for="model" class="font-semibold w-24">Model</label>
            <p-select id="model" disabled [(ngModel)]="model" [options]="models" optionLabel="label" optionValue="value" />
          </div>
          <span class="text-secondary">Select your preferred model for similarity scoring.</span>
          <p-divider />
          
          <div class="flex justify-end gap-2 mt-4">
            <p-button 
              label="Cancel" 
              severity="secondary" 
              (click)="visible = false" 
            />
            <p-button 
              label="Save" 
              (click)="saveConfig()" 
            />
          </div>
        </div>
      </p-dialog>
      <div class="content-container">
        <router-outlet></router-outlet>
      </div>
    </div>
  `,
  styles: [`
    .app-container {
      min-height: 100vh;
    }
    .content-container {
      margin-top: 72px;
    }
    .user-menu {
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }
    .username-text {
      font-weight: 500;
      color: var(--text-color);
      font-size: 0.9rem;
    }
    :host ::ng-deep {
      .p-dialog-header {
        padding-bottom: 1rem;
      }

      .p-dialog-content {
        padding: 1.5rem;
      }
    }
  `]
})
export class AppComponent {
  visible: boolean = false;
  username: string = '';
  model: string = '';
  models: { label: string, value: string }[] = [];
  items: MenuItem[] = [
    { label: 'Home', icon: 'pi pi-home', routerLink: '/' },
    { label: 'Programs', icon: 'pi pi-video', routerLink: '/programs' },
  ];

  constructor(private userService: UserService, private modelService: ModelService) {
    this.username = this.userService.getUsername();
    this.model = this.modelService.getModel();
    this.models = this.modelService.getOptions();
    if (!this.username) {
      this.showDialog();
    }
  }


  showDialog() {
    this.username = this.userService.getUsername();
    this.visible = true;
  }

  saveConfig() {
    if (this.username.trim()) {
      this.userService.setUsername(this.username);
      this.modelService.setModel(this.model);
      this.visible = false;
    }
  }

  navigateToGitHub() {
    window.open('https://github.com/MG-Guan/audio-llm-hackathon', '_blank');
  }
}