'use client';

import React from 'react';
import { Message } from '@/types';
import { Citation } from './Citation';
import { User, Bot, AlertCircle } from 'lucide-react';

interface MessageBubbleProps {
  message: Message;
}

export function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === 'user';

  return (
    <div
      className={`flex gap-4 ${isUser ? 'flex-row-reverse' : ''}`}
    >
      <div
        className={`flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center ${
          isUser ? 'bg-blue-500' : 'bg-purple-500'
        }`}
      >
        {isUser ? (
          <User className="w-5 h-5 text-white" />
        ) : (
          <Bot className="w-5 h-5 text-white" />
        )}
      </div>

      <div className={`flex-1 max-w-3xl ${isUser ? 'text-right' : ''}`}>
        <div className="flex items-center gap-2 mb-1">
          <span className="font-semibold text-sm text-gray-900">
            {isUser ? 'You' : 'Assistant'}
          </span>
          <span className="text-xs text-gray-500">
            {message.timestamp.toLocaleTimeString([], {
              hour: '2-digit',
              minute: '2-digit',
            })}
          </span>
        </div>

        <div
          className={`inline-block text-left rounded-2xl px-5 py-3 ${
            isUser
              ? 'bg-blue-500 text-white'
              : 'bg-gray-100 text-gray-900'
          }`}
        >
          {message.content ? (
            <p className="text-sm leading-relaxed whitespace-pre-wrap">
              {message.content}
            </p>
          ) : message.isStreaming ? (
            <div className="flex items-center gap-1 py-2">
              <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
              <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
              <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
            </div>
          ) : null}
        </div>

        {!isUser && message.citations && message.citations.length > 0 && (
          <div className="mt-3 space-y-2">
            <p className="text-xs font-medium text-gray-500 uppercase tracking-wide">
              Sources ({message.citations.length})
            </p>
            <div className="space-y-2">
              {message.citations.map((citation, index) => (
                <Citation
                  key={citation.id}
                  citation={citation}
                  index={index}
                />
              ))}
            </div>
          </div>
        )}

        {!isUser && message.content.startsWith('Error:') && (
          <div className="mt-2 flex items-center gap-2 text-red-600 text-sm">
            <AlertCircle className="w-4 h-4" />
            <span>Failed to generate response</span>
          </div>
        )}
      </div>
    </div>
  );
}
