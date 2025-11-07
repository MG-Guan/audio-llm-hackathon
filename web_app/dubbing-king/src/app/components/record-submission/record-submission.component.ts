import { Component, OnInit, ViewChild, ElementRef, OnDestroy } from '@angular/core';
import { ActivatedRoute, Router } from '@angular/router';
import { CommonModule } from '@angular/common';
import { Program } from '../../models/program.model';
import { ProgramService } from '../../services/program.service';
import { switchMap } from 'rxjs/operators';
import { TagModule } from 'primeng/tag';
import { DividerModule } from 'primeng/divider';
import { ButtonModule } from 'primeng/button';
import { DialogModule } from 'primeng/dialog';
import { RecordingState, RecordingStatus } from '../../models/recording-state.model';
import { SubmissionService } from '../../services/submission.service';
import { Submission } from '../../models/submission.model';
import { UserService } from '../../services/user.service';
import { ModelService } from '../../services/model.service';
import config from '../../services/config';
import { SpeakerSeverityService } from '../../services/speaker-severity.service';

declare var grecaptcha: {
  render: (container: HTMLElement, params: {
    sitekey: string;
    callback: (token: string) => void;
    'error-callback'?: () => void;
    'expired-callback'?: () => void;
  }) => number;
  reset: (widgetId?: number) => void;
};

@Component({
  selector: 'app-record-submission',
  standalone: true,
  imports: [CommonModule, TagModule, DividerModule, ButtonModule, DialogModule],
  styleUrls: ['./record-submission.component.scss'],
  template: `
    <div class="record-submission" *ngIf="program">
      <div class="video-container">
        <div class="video-player">
          <video playsinline [src]="program.videoUrl" controls #videoPlayer></video>
        </div>
      </div>
      
      <div class="transcripts">
        <div class="transcript-list">
          <div *ngFor="let clip of program.clips" class="transcript-item">
            <div class="transcript-info">
              <p-tag value="{{clip.speaker || 'Speaker'}}" [severity]="getSeverityForSpeaker(clip.speaker)"></p-tag>
              <div class="clip-duration">{{formatDuration(clip.duration)}}</div>
            </div>
            <div class="transcript-text-container">
              <p class="transcript-text text-lg">{{clip.transcript}}</p>
            </div>
          </div>
        </div>
      </div>

      <div class="recording-section">
        <h1 class="title text-lg font-bold">{{program.title}}</h1>
        <p-divider></p-divider>
        <div class="recording-controls">
          <div class="recording-status" [ngSwitch]="recordingStatus.state">
            <ng-container *ngSwitchCase="RecordingState.IDLE">
              <div class="status-text">Ready to Record</div>
            </ng-container>
            <ng-container *ngSwitchCase="RecordingState.RECORDING">
              <div class="status-text recording">Recording...</div>
              <div class="recording-time">{{recordingStatus.time}}</div>
            </ng-container>
            <ng-container *ngSwitchCase="RecordingState.RECORDED">
              <div class="status-text">Recording Complete</div>
              <div class="recording-time">{{recordingStatus.time}}</div>
            </ng-container>
            <ng-container *ngSwitchCase="RecordingState.PLAYING">
              <div class="status-text">Playing Recording...</div>
              <div class="recording-time">{{recordingStatus.time}}</div>
            </ng-container>
            <ng-container *ngSwitchCase="RecordingState.SUBMITTING">
              <div class="status-text">Submitting...</div>
            </ng-container>
          </div>
          
          <div class="controls" [ngSwitch]="recordingStatus.state">
            <!-- IDLE State -->
            <button 
              *ngSwitchCase="RecordingState.IDLE"
              pButton 
              class="p-button-rounded p-button-danger record-button" 
              icon="pi pi-microphone" 
              (click)="startRecording()"
            ></button>
            
            <!-- RECORDING State -->
            <button 
              *ngSwitchCase="RecordingState.RECORDING"
              pButton 
              class="p-button-rounded p-button-danger" 
              icon="pi pi-stop" 
              (click)="stopRecording()"
            ></button>

            <!-- RECORDED State -->
            <div *ngSwitchCase="RecordingState.RECORDED" class="playback-controls">
              <button 
                pButton 
                class="p-button-rounded p-button-secondary" 
                icon="pi pi-refresh" 
                (click)="retryRecording()"
              ></button>
              <button 
                pButton 
                class="p-button-rounded p-button-secondary" 
                icon="pi pi-play" 
                (click)="playRecording()"
              ></button>
              <button 
                pButton 
                class="p-button-rounded p-button-success" 
                icon="pi pi-check" 
                (click)="submitRecording()"
              ></button>
            </div>

            <!-- PLAYING State -->
            <div *ngSwitchCase="RecordingState.PLAYING" class="playback-controls">
              <button 
                pButton 
                class="p-button-rounded p-button-secondary" 
                icon="pi pi-stop" 
                (click)="stopPlayback()"
              ></button>
            </div>

            <!-- SUBMITTING State -->
            <div *ngSwitchCase="RecordingState.SUBMITTING" class="submitting">
              <i class="pi pi-spin pi-spinner"></i>
            </div>
          </div>
        </div>
      </div>

      <!-- reCAPTCHA Dialog -->
      <p-dialog 
        [(visible)]="showRecaptchaDialog" 
        [modal]="true" 
        [closable]="false"
        [draggable]="false"
        [resizable]="false"
        styleClass="recaptcha-dialog"
        [style]="{width: '400px'}"
      >
        <ng-template #headless>
          <div class="recaptcha-container">
            <h3 class="text-md font-bold">Please verify you're human</h3>
            <div class="g-recaptcha" [attr.data-sitekey]="recaptchaSiteKey" #recaptchaContainer></div>
          </div>
        </ng-template>
      </p-dialog>
    </div>
  `
})
export class RecordSubmissionComponent implements OnInit, OnDestroy {
  @ViewChild('videoPlayer') videoPlayer!: ElementRef<HTMLVideoElement>;
  @ViewChild('recaptchaContainer') recaptchaContainer!: ElementRef<HTMLDivElement>;
  program?: Program;
  RecordingState = RecordingState; // For template access
  recordingStatus: RecordingStatus = {
    state: RecordingState.IDLE,
    time: '00:00'
  };
  showRecaptchaDialog = false;
  recaptchaSiteKey = config.recaptchaSiteKey;
  recaptchaWidgetId?: number;

  private recordingInterval?: number;
  private startTime?: number;
  private mediaRecorder?: MediaRecorder;
  private audioChunks: Blob[] = [];
  private audioPlayer?: HTMLAudioElement;

  constructor(
    private route: ActivatedRoute,
    private programService: ProgramService,
    private submissionService: SubmissionService,
    private router: Router,
    private userService: UserService,
    private modelService: ModelService,
    private speakerSeverityService: SpeakerSeverityService
  ) {}

  ngOnInit() {
    this.route.params.pipe(
      switchMap(params => this.programService.getProgram(params['id']))
    ).subscribe(program => {
      this.program = program;
    });
  }

  private clearRecaptcha(): void {
    if (this.recaptchaContainer?.nativeElement) {
      // Clear the container HTML
      this.recaptchaContainer.nativeElement.innerHTML = '';
    }
    
    // Reset and clear widget ID if it exists
    if (this.recaptchaWidgetId !== undefined && typeof grecaptcha !== 'undefined' && grecaptcha.reset) {
      try {
        grecaptcha.reset(this.recaptchaWidgetId);
      } catch (e) {
        // Widget might already be destroyed, ignore
      }
      this.recaptchaWidgetId = undefined;
    }
  }

  private renderRecaptcha(): Promise<string> {
    return new Promise((resolve, reject) => {
      if (typeof grecaptcha === 'undefined' || !grecaptcha.render) {
        reject(new Error('reCAPTCHA not loaded'));
        return;
      }

      if (!this.recaptchaContainer?.nativeElement) {
        reject(new Error('reCAPTCHA container not found'));
        return;
      }

      // Clear any existing widget before rendering a new one
      this.clearRecaptcha();

      // Render new widget using standard API
      try {
        this.recaptchaWidgetId = grecaptcha.render(this.recaptchaContainer.nativeElement, {
          sitekey: this.recaptchaSiteKey,
          callback: (token: string) => {
            resolve(token);
          },
          'error-callback': () => {
            reject(new Error('reCAPTCHA verification failed'));
          },
          'expired-callback': () => {
            reject(new Error('reCAPTCHA token expired'));
          }
        });
      } catch (error: any) {
        // If widget is already rendered, try to handle it
        if (error?.message?.includes('already been rendered')) {
          // Clear and try again
          this.clearRecaptcha();
          try {
            this.recaptchaWidgetId = grecaptcha.render(this.recaptchaContainer.nativeElement, {
              sitekey: this.recaptchaSiteKey,
              callback: (token: string) => {
                resolve(token);
              },
              'error-callback': () => {
                reject(new Error('reCAPTCHA verification failed'));
              },
              'expired-callback': () => {
                reject(new Error('reCAPTCHA token expired'));
              }
            });
          } catch (retryError) {
            reject(new Error(`Failed to render reCAPTCHA: ${retryError}`));
          }
        } else {
          reject(new Error(`Failed to render reCAPTCHA: ${error}`));
        }
      }
    });
  }

  private pauseVideo() {
    if (this.videoPlayer?.nativeElement) {
      this.videoPlayer.nativeElement.pause();
    }
  }

  async startRecording() {
    try {
      this.pauseVideo();
      
      // Safari iOS specific constraints
      const constraints = {
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          sampleRate: 44100
        }
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      
      // Use different MIME types based on browser support
      const mimeType = [
        'audio/webm',
        'audio/mp4',
        'audio/ogg',
        'audio/wav'
      ].find(type => MediaRecorder.isTypeSupported(type)) || '';

      this.mediaRecorder = new MediaRecorder(stream, {
        mimeType: mimeType || undefined,
        audioBitsPerSecond: 128000
      });
      
      this.audioChunks = [];

      this.mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          this.audioChunks.push(event.data);
        }
      };

      this.mediaRecorder.onstop = () => {
        const audio = new Blob(this.audioChunks, { type: mimeType || 'audio/webm' });
        stream.getTracks().forEach(track => track.stop());
        
        this.recordingStatus = {
          state: RecordingState.RECORDED,
          time: this.recordingStatus.time,
          audio
        };
      };

      // Request data every second for Safari iOS
      this.mediaRecorder.start(1000);
      this.startTime = Date.now();
      this.updateRecordingTime();
      
      this.recordingStatus = {
        state: RecordingState.RECORDING,
        time: '00:00'
      };
    } catch (err) {
      console.error('Error accessing microphone:', err);
      alert('Failed to access microphone. Please ensure you have microphone device and granted microphone permissions.');
    }
  }

  stopRecording() {
    if (this.mediaRecorder?.state === 'recording') {
      this.mediaRecorder.stop();
      clearInterval(this.recordingInterval);
      // Update UI state immediately
      this.recordingStatus = {
        ...this.recordingStatus,
        state: RecordingState.RECORDED
      };
    }
  }

  retryRecording() {
    this.recordingStatus = {
      state: RecordingState.IDLE,
      time: '00:00'
    };
  }

  playRecording() {
    const audio = this.recordingStatus.audio;
    if (audio instanceof Blob) {
      this.pauseVideo();
      this.recordingStatus = {
        ...this.recordingStatus,
        state: RecordingState.PLAYING
      };

      this.audioPlayer = new Audio(URL.createObjectURL(audio));
      this.audioPlayer.onended = () => {
        this.recordingStatus = {
          ...this.recordingStatus,
          state: RecordingState.RECORDED
        };
      };
      this.audioPlayer.play();
    }
  }

  stopPlayback() {
    if (this.audioPlayer) {
      this.audioPlayer.pause();
      this.audioPlayer = undefined;
      this.recordingStatus = {
        ...this.recordingStatus,
        state: RecordingState.RECORDED
      };
    }
  }

  async submitRecording() {
    const audio = this.recordingStatus.audio;
    if (audio instanceof Blob) {
      this.showRecaptchaDialog = true;
      
      // Wait for dialog to be rendered and reCAPTCHA to be ready
      try {
        await this.waitForRecaptchaReady();
        const token = await this.renderRecaptcha();
        await this.performSubmission(audio, token);
      } catch (error) {
        console.error('reCAPTCHA error:', error);
        alert('reCAPTCHA verification failed. Please try again.');
        this.clearRecaptcha();
        this.showRecaptchaDialog = false;
      }
    } else {
      alert('No recording to submit');
    }
  }

  private async waitForRecaptchaReady(): Promise<void> {
    const maxAttempts = 100; // 10 seconds max wait time
    let attempts = 0;

    return new Promise((resolve, reject) => {
      const checkReady = () => {
        attempts++;
        
        // Check if grecaptcha is loaded
        if (typeof grecaptcha === 'undefined' || !grecaptcha.render) {
          if (attempts >= maxAttempts) {
            reject(new Error('reCAPTCHA script failed to load'));
            return;
          }
          setTimeout(checkReady, 100);
          return;
        }

        // Check if dialog container is available
        if (!this.recaptchaContainer?.nativeElement) {
          if (attempts >= maxAttempts) {
            reject(new Error('reCAPTCHA container not found'));
            return;
          }
          setTimeout(checkReady, 100);
          return;
        }

        // Check if container is available and in the DOM
        const container = this.recaptchaContainer.nativeElement;
        
        // Check if element is in the DOM and has a parent (dialog is rendered)
        if (!container.parentElement) {
          if (attempts >= maxAttempts) {
            reject(new Error('reCAPTCHA container not attached to DOM'));
            return;
          }
          setTimeout(checkReady, 100);
          return;
        }

        // Check if element is not explicitly hidden
        const computedStyle = window.getComputedStyle(container);
        if (computedStyle.display === 'none' || computedStyle.visibility === 'hidden') {
          if (attempts >= maxAttempts) {
            reject(new Error('reCAPTCHA container is hidden'));
            return;
          }
          setTimeout(checkReady, 100);
          return;
        }

        // Element is ready - wait for browser to render dialog
        // Use requestAnimationFrame to ensure dialog is painted
        requestAnimationFrame(() => {
          requestAnimationFrame(() => {
            // Give it a moment for dialog animation to complete
            setTimeout(resolve, 300);
          });
        });
      };

      // Start checking after Angular change detection cycle
      setTimeout(checkReady, 50);
    });
  }

  private async performSubmission(audio: Blob, token: string) {
    this.showRecaptchaDialog = false;
    this.recordingStatus = {
      ...this.recordingStatus,
      state: RecordingState.SUBMITTING
    };

    try {
      const submission = await this.submissionService.postSubmission({
        programId: this.program?.id ?? '',
        clipId: this.program?.clips[0].id ?? '',
        username: this.userService.getUsername(),
        audio: audio,
        model: this.modelService.getModel(),
        token: token
      });
      this.router.navigate(['/submission', submission.id]);
    } catch (error) {
      alert('Failed to submit recording. Please try again.');
      // Return to recorded state so user can retry
      this.recordingStatus = {
        ...this.recordingStatus,
        state: RecordingState.RECORDED,
        audio: audio
      };
    } finally {
      // Clear reCAPTCHA widget when submission completes (success or failure)
      this.clearRecaptcha();
    }
  }

  private updateRecordingTime() {
    this.recordingInterval = window.setInterval(() => {
      if (this.startTime) {
        const elapsed = Date.now() - this.startTime;
        const seconds = Math.floor(elapsed / 1000);
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = seconds % 60;
        const time = `${minutes.toString().padStart(2, '0')}:${remainingSeconds.toString().padStart(2, '0')}`;
        
        this.recordingStatus = {
          ...this.recordingStatus,
          time
        };
      }
    }, 1000);
  }

  formatDuration(seconds: number): string {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
  }

  getSeverityForSpeaker(speaker: string | undefined): 'secondary' | 'success' | 'info' | 'warn' | 'danger' {
    return this.speakerSeverityService.getSeverityForSpeaker(speaker);
  }

  ngOnDestroy() {
    if (this.recordingInterval) {
      clearInterval(this.recordingInterval);
    }
    if (this.mediaRecorder?.state === 'recording') {
      this.mediaRecorder.stop();
    }
    if (this.audioPlayer) {
      this.audioPlayer.pause();
    }
    // Clear reCAPTCHA widget
    this.clearRecaptcha();
  }
}