import { Injectable } from '@angular/core';
import { CookieService } from './cookie.service';
import { UsernameGeneratorService } from './username-generator.service';

@Injectable({
  providedIn: 'root'
})
export class UserService {
  private readonly USERNAME_KEY = 'username';

  constructor(
    private cookieService: CookieService,
    private usernameGenerator: UsernameGeneratorService
  ) {}

  getUsername(): string {
    const savedUsername = this.cookieService.getCookie(this.USERNAME_KEY, '');
    if (!savedUsername) {
      // No username found, generate and save a new one
      const newUsername = this.usernameGenerator.generateUsername();
      this.setUsername(newUsername);
      return newUsername;
    }
    return savedUsername;
  }

  setUsername(username: string): void {
    this.cookieService.setCookie(this.USERNAME_KEY, username);
  }

  /**
   * Generates a new random username without saving it
   */
  generateNewUsername(): string {
    return this.usernameGenerator.generateUsername();
  }
}
