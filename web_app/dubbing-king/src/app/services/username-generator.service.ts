import { Injectable } from '@angular/core';

@Injectable({
  providedIn: 'root'
})
export class UsernameGeneratorService {
  private readonly adjectives = [
    'Happy', 'Clever', 'Brave', 'Gentle', 'Swift', 'Bright', 'Noble', 'Wise',
    'Calm', 'Eager', 'Kind', 'Merry', 'Proud', 'Quick', 'Silent', 'Warm',
    'Witty', 'Bold', 'Jolly', 'Lively'
  ];

  private readonly nouns = [
    'Panda', 'Eagle', 'Tiger', 'Dolphin', 'Fox', 'Wolf', 'Bear', 'Lion',
    'Owl', 'Dragon', 'Phoenix', 'Falcon', 'Hawk', 'Deer', 'Horse', 'Rabbit',
    'Koala', 'Penguin', 'Otter', 'Lynx'
  ];

  generateUsername(): string {
    const adjective = this.adjectives[Math.floor(Math.random() * this.adjectives.length)];
    const noun = this.nouns[Math.floor(Math.random() * this.nouns.length)];
    const number = Math.floor(Math.random() * 999) + 1; // 1-999
    
    return `${adjective}${noun}${number}`;
  }
}
