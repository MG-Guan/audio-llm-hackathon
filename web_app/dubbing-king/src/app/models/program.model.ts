import { Clip } from './clip.model';

export interface Program {
  id: string;
  title: string;
  clips: Clip[];
  thumbnailUrl: string;
  description: string;
  duration: number;
  tags: string[];
  videoUrl: string;
}
