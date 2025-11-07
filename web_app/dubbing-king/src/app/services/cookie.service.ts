import { Injectable } from '@angular/core';

@Injectable({
  providedIn: 'root'
})
export class CookieService {
  private readonly COOKIE_PREFIX = 'dubbing_king_';

  setCookie(key: string, value: string, expiryYears: number = 1): void {
    const sanitizedKey = this.sanitizeKey(key);
    const expiry = new Date();
    expiry.setFullYear(expiry.getFullYear() + expiryYears);
    
    const cookie = `${this.COOKIE_PREFIX}${sanitizedKey}=${encodeURIComponent(value)}; expires=${expiry.toUTCString()}; path=/; SameSite=Strict`;
    document.cookie = cookie;
  }

  getCookie(key: string, defaultValue: string): string {
    const sanitizedKey = this.sanitizeKey(key);
    const cookieKey = `${this.COOKIE_PREFIX}${sanitizedKey}`;
    
    const value = document.cookie
      .split('; ')
      .find(row => row.startsWith(`${cookieKey}=`))
      ?.split('=')[1];

    return value ? decodeURIComponent(value) : defaultValue;
  }

  private sanitizeKey(key: string): string {
    return key.replace(/[^a-zA-Z0-9_]/g, '_');
  }
}
