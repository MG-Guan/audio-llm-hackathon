import { Injectable } from '@angular/core';

@Injectable({
  providedIn: 'root'
})
export class SpeakerSeverityService {
  private readonly severityOptions = ['secondary', 'success', 'info', 'warn', 'danger'] as const;

  getSeverityForSpeaker(speaker: string | undefined): 'secondary' | 'success' | 'info' | 'warn' | 'danger' {
    const speakerName = speaker || 'Speaker';
    
    // Simple hash function to convert string to number
    let hash = 0;
    for (let i = 0; i < speakerName.length; i++) {
      const char = speakerName.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    
    // Use absolute value and modulo to get index in severity array
    const index = Math.abs(hash) % this.severityOptions.length;
    return this.severityOptions[index];
  }
}

