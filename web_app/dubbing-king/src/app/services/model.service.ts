import { Injectable } from '@angular/core';
import { CookieService } from './cookie.service';

const models = {
    'options': [
        "whisper-medium",
        "whisper-medium_wavlm-large",
        "higgs-audio-understanding"
    ],
    default: "whisper-medium_wavlm-large"
}

@Injectable({
  providedIn: 'root'
})
export class ModelService {
  private readonly MODEL_KEY = 'model';
  private readonly DEFAULT_MODEL = models.default;

  constructor(private cookieService: CookieService) {}

  getModel(): string {
    return this.cookieService.getCookie(this.MODEL_KEY, this.DEFAULT_MODEL);
  }

  setModel(model: string): void {
    this.cookieService.setCookie(this.MODEL_KEY, model);
  }

  getOptions(): { label: string, value: string }[] {
    return models.options.map(model => ({ label: model, value: model }));
  }
}
