import { format } from "date-fns";

const UTC_TIMESTAMP = /^(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2}:\d{2})(?:\.\d+)?$/;

function parseImageTime(value: string): Date | null {
  const match = UTC_TIMESTAMP.exec(value);
  const date = new Date(match ? `${match[1]}T${match[2]}Z` : value);
  return Number.isNaN(date.getTime()) ? null : date;
}

export function formatImageLocalTime(value: string): string {
  const date = parseImageTime(value);
  return date ? format(date, "yyyy-MM-dd HH:mm:ss") : value;
}

export function imageLocalDate(value: string, fallback: string): string {
  const date = parseImageTime(value);
  return date ? format(date, "yyyy-MM-dd") : fallback;
}
