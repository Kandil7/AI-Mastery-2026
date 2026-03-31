'use client';

import React, { useState } from 'react';
import { Citation as CitationType } from '@/types';
import { ChevronDown, ChevronUp, FileText, ExternalLink } from 'lucide-react';

interface CitationProps {
  citation: CitationType;
  index: number;
}

export function Citation({ citation, index }: CitationProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  const getScoreColor = (score: number) => {
    if (score >= 0.9) return 'bg-green-100 text-green-800';
    if (score >= 0.7) return 'bg-yellow-100 text-yellow-800';
    return 'bg-red-100 text-red-800';
  };

  return (
    <div className="border border-gray-200 rounded-lg overflow-hidden bg-white">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full px-4 py-3 flex items-center justify-between hover:bg-gray-50 transition-colors"
      >
        <div className="flex items-center gap-3">
          <span className="flex-shrink-0 w-6 h-6 rounded-full bg-blue-100 text-blue-600 text-xs font-semibold flex items-center justify-center">
            {index + 1}
          </span>
          <div className="flex items-center gap-2">
            <FileText className="w-4 h-4 text-gray-500" />
            <span className="font-medium text-gray-900 text-sm">
              {citation.documentName}
            </span>
            {citation.pageNumber && (
              <span className="text-xs text-gray-500">
                Page {citation.pageNumber}
              </span>
            )}
          </div>
        </div>
        <div className="flex items-center gap-2">
          <span
            className={`text-xs px-2 py-1 rounded-full font-medium ${getScoreColor(
              citation.score
            )}`}
          >
            {(citation.score * 100).toFixed(0)}% match
          </span>
          {isExpanded ? (
            <ChevronUp className="w-4 h-4 text-gray-400" />
          ) : (
            <ChevronDown className="w-4 h-4 text-gray-400" />
          )}
        </div>
      </button>
      
      {isExpanded && (
        <div className="px-4 pb-4 pt-2 bg-gray-50 border-t border-gray-200">
          <div className="flex items-start gap-2">
            <div className="flex-1">
              <p className="text-sm text-gray-700 leading-relaxed">
                "{citation.text}"
              </p>
              <div className="mt-2 flex items-center gap-4 text-xs text-gray-500">
                <span>Chunk {citation.chunkIndex}</span>
                <span>ID: {citation.id}</span>
              </div>
            </div>
            <button
              className="flex-shrink-0 p-2 hover:bg-white rounded-lg transition-colors"
              title="View in document"
            >
              <ExternalLink className="w-4 h-4 text-gray-400" />
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
