import { Injectable } from '@angular/core';
import { Observable, of } from 'rxjs';
import { Program } from '../models/program.model';
import config from './config';

@Injectable({
  providedIn: 'root'
})
export class ProgramService {
  private mockPrograms: Program[] = [
    {
      id: 'Blue-Jays-Epic-1993[part01]',
      title: 'Blue Jays 1993 World Series: Broadcast',
      description: 'The Blue Jays defeated the Phillies in six games, becoming the seventh franchise in MLB history to win back-to-back championships.',
      thumbnailUrl: `${config.staticUrl}/Blue-Jays-Epic-1993_part01.jpg`,
      duration: 18,
      tags: ['Broadcast', 'Trending'],
      videoUrl: `${config.staticUrl}/Blue-Jays-Epic-1993_part01.mov`,
      clips: [
        {
          id: '1',
          duration: 18,
          transcript: 'The Blue Jays are World Series Champions, as Joe Carter hits a three-run home run in the ninth inning and the Blue Jays have repeated as World Series champions!',
          audioUrl: `${config.staticUrl}/Blue-Jays-Epic-1993_part01.wav`,
          speaker: 'Tom Cheek',
        },
        {
          id: '2',
          duration: 2,
          transcript: 'Touch \'em all, Joe, you\'ll never hit a bigger home run in your life!',
          speaker: 'Tom Cheek',
        }
      ],
    },
    {
      id: 'PeppaPig-s06e02[part01]',
      title: 'Peppa Pig - S6E2: Voice-over',
      description: 'Introduction to the story where Captain Dog is talking about his adventures.',
      thumbnailUrl: `${config.staticUrl}/PeppaPig-s06e02_part01.mov_snapshot_00.02.061.jpg`,
      duration: 9,
      tags: ['Voice-over', 'Beginner'],
      videoUrl: `${config.staticUrl}/PeppaPig-s06e02_part01.mov`,
      clips: [
        {
          id: '1',
          duration: 3,
          transcript: 'Desert Island!',
          audioUrl: `${config.staticUrl}/PeppaPig-s06e02_part01.wav`,
          speaker: 'Speaker 1',
        },
        {
          id: '2',
          duration: 6,
          transcript: 'Peppa and George are at Danny Dog\'s house. Captain Dog is telling stories of when he was a sailor.',
          speaker: 'Speaker 2',
        }
      ],
    },
    {
      id: 'PeppaPig-s06e02[part02]',
      title: 'Peppa Pig - S6E2: Monologue',
      description: 'Captain Dog is telling stories of when he was a sailor.',
      thumbnailUrl: `${config.staticUrl}/PeppaPig-s06e02_part02.mov_snapshot_00.06.457.jpg`,
      duration: 12,
      tags: ['Male Voice', 'Beginner'],
      videoUrl: `${config.staticUrl}/PeppaPig-s06e02_part02.mov`,
      clips: [
        {
          id: '1',
          duration: 12,
          transcript: 'I sailed all around the world and then I came home again. But now I\'m back for good. I\'ll never get on a boat again.',
          audioUrl: `${config.staticUrl}/PeppaPig-s06e02_part02.wav`,
          speaker: 'Captain Dog',
        },
      ],
    },
    {
      id: 'PeppaPig-s06e02[part03]',
      title: 'Peppa Pig - S6E2: Conversation',
      description: 'Grandad Dog, Granpa Pig and Grumpy Rabbit invite Captain Dog to go on a fishing trip.',
      thumbnailUrl: `${config.staticUrl}/PeppaPig-s06e02_part03.mov_snapshot_00.07.382.jpg`,
      duration: 30,
      tags: ['Multi Role', 'Complex'],
      videoUrl: `${config.staticUrl}/PeppaPig-s06e02_part03.mov`,
      clips: [
        {
          id: '1',
          duration: 2,
          transcript: 'Daddy, do you miss the sea?',
          audioUrl: `${config.staticUrl}/PeppaPig-s06e02_part03.wav`,
          speaker: 'Danny Dog',
        },
        {
          id: '2',
          duration: 3,
          transcript: 'Well, sometimes.',
          speaker: 'Captain Dog',
        },
        {
          id: '3',
          duration: 5,
          transcript: 'It is Grandad Dog, Grandpa Pig and Grumpy Rabbit. Hello!',
          speaker: 'Speaker 3',
        },
        {
          id: '4',
          duration: 3,
          transcript: 'Can Captain Dog come out to play?',
          speaker: 'Grandad Dog',
        },
        {
          id: '5',
          duration: 1,
          transcript: 'What?',
          speaker: 'Captain Dog',
        },
        {
          id: '6',
          duration: 5,
          transcript: ' We are going on a fishing trip! On a boat! On the sea!',
          speaker: 'Multiple',
        },
        {
          id: '7',
          duration: 1,
          transcript: 'OK, let\'s go!',
          speaker: 'Captain Dog',
        },
        {
          id: '8',
          duration: 5,
          transcript: 'But Daddy, you said you\'d never get on a boat again.',
          speaker: 'Danny Dog',
        },
        {
          id: '9',
          duration: 2,
          transcript: 'Oh yes, so I did.',
          speaker: 'Captain Dog',
        },
        {
          id: '10',
          duration: 1,
          transcript: 'OK, bye bye!',
          speaker: 'Multiple',
        },
        {
          id: '11',
          duration: 1,
          transcript: 'Bye!',
          speaker: 'Captain Dog',
        },
      ],
    },
    {
      id: 'PeppaPig-s06e02[part04]',
      title: 'Peppa Pig - S6E2: Conversation 2',
      description: 'Captain Dog rejects the invitation to go on a fishing trip. What will he do now?',
      thumbnailUrl: `${config.staticUrl}/PeppaPig-s06e02_part04.mov_snapshot_00.18.533.jpg`,
      duration: 18,
      tags: ['Multi Role', 'Intermediate'],
      videoUrl: `${config.staticUrl}/PeppaPig-s06e02_part04.mov`,
      clips: [
        {
          id: '1',
          duration: 6,
          transcript: 'Well, there they go, off on a boat, without me.',
          audioUrl: `${config.staticUrl}/PeppaPig-s06e02_part04.wav`,
          speaker: 'Captain Dog',
        },
        {
          id: '2',
          duration: 2,
          transcript: 'What are you going to do now, Daddy?',
          speaker: 'Danny Dog',
        },
        {
          id: '3',
          duration: 3,
          transcript: 'Oh, I don\'t know. Maybe I\'ll clean the seaweed off the house.',
          speaker: 'Captain Dog',
        },
        {
          id: '4',
          duration: 3,
          transcript: 'You don\'t get seaweeds on houses.',
          speaker: 'Peppa Pig',
        },
        {
          id: '5',
          duration: 2,
          transcript: 'No, of course not.',
          speaker: 'Captain Dog',
        },
      ],
    }
  ];

  getPrograms(): Observable<Program[]> {
    return of(this.mockPrograms);
  }

  getProgram(id: string): Observable<Program | undefined> {
    return of(this.mockPrograms.find(p => p.id === id));
  }
}
