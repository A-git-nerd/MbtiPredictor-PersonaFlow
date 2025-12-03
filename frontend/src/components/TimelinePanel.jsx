import React, { useState } from 'react';
import GenderSelectionModal from './GenderSelectionModal';
import MBTIPopup from './MBTIPopup';

const TimelinePanel = ({ timeline, selectedUser, onClose }) => {
    const [gender, setGender] = useState(null);
    const [selectedMbti, setSelectedMbti] = useState(null);

    const handleCardClick = (item) => {
        setSelectedMbti({
            type: item.dominant_mbti,
            sampleMessages: item.sample_messages,
            allMessages: item.all_messages
        });
    };

    const handleClosePopup = () => {
        setSelectedMbti(null);
    };

    // Filter out Unknown entries
    const validTimeline = timeline.filter(item => item.dominant_mbti !== 'Unknown');

    return (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex justify-center items-center p-4 z-50 animate-fade-in">
            <div className="bg-white rounded-2xl shadow-2xl w-full max-w-5xl max-h-[90vh] flex flex-col overflow-hidden animate-slide-up relative">

                {/* Header */}
                <div className="p-6 border-b border-gray-100 flex justify-between items-center bg-gradient-to-r from-gray-50 to-white">
                    <div>
                        <h2 className="text-2xl font-bold text-gray-800">Personality Timeline</h2>
                        <p className="text-sm text-gray-500">Analysis for <span className="font-semibold text-pink-600">{selectedUser}</span></p>
                    </div>
                    <button
                        onClick={onClose}
                        className="w-10 h-10 rounded-full bg-gray-100 hover:bg-gray-200 flex items-center justify-center text-gray-600 transition-colors"
                    >
                        ✕
                    </button>
                </div>

                {/* Content */}
                <div className="p-8 overflow-y-auto flex-1 bg-gray-50/50">
                    {validTimeline.length === 0 ? (
                        <div className="text-center py-20">
                            <p className="text-gray-500 text-lg">No sufficient data available to generate a timeline for this user.</p>
                        </div>
                    ) : (
                        <div className="relative max-w-3xl mx-auto">
                            {/* Vertical Line */}
                            <div className="absolute left-8 top-0 bottom-0 w-0.5 bg-gradient-to-b from-pink-300 via-purple-300 to-blue-300"></div>

                            <div className="space-y-12">
                                {validTimeline.map((item, index) => (
                                    <div
                                        key={index}
                                        className="relative flex items-start gap-8 group cursor-pointer"
                                        onClick={() => handleCardClick(item)}
                                    >
                                        {/* Node */}
                                        <div className="absolute left-[26px] mt-1.5 w-4 h-4 rounded-full bg-white border-4 border-pink-500 z-10 group-hover:scale-125 transition-transform shadow-sm"></div>

                                        {/* Card */}
                                        <div className="ml-16 flex-1 bg-white p-6 rounded-xl shadow-sm border border-gray-100 hover:shadow-md transition-all hover:-translate-y-1">
                                            <div className="flex flex-wrap justify-between items-start gap-4 mb-3">
                                                <div className="flex items-center gap-2 text-sm text-gray-500 font-medium">
                                                    <span className="bg-gray-100 px-2 py-1 rounded text-xs">📅 {item.start_date}</span>
                                                    <span>→</span>
                                                    <span className="bg-gray-100 px-2 py-1 rounded text-xs">📅 {item.end_date}</span>
                                                </div>
                                                <span className={`px-4 py-1.5 rounded-full text-sm font-bold shadow-sm ${getMbtiColor(item.dominant_mbti)}`}>
                                                    {item.dominant_mbti}
                                                </span>
                                            </div>

                                            <div className="flex items-center gap-4">
                                                <div className="text-4xl">{getMbtiIcon(item.dominant_mbti)}</div>
                                                <div>
                                                    <h3 className="font-bold text-gray-800 text-lg">
                                                        {getMbtiTitle(item.dominant_mbti)}
                                                    </h3>
                                                    <p className="text-sm text-gray-600 mt-1">
                                                        Dominant traits detected in this period.
                                                    </p>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}
                </div>
            </div>

            {/* Modals */}
            {selectedMbti && !gender && (
                <GenderSelectionModal
                    onSelect={setGender}
                    onClose={handleClosePopup}
                />
            )}

            {selectedMbti && gender && (
                <MBTIPopup
                    mbti={selectedMbti.type}
                    gender={gender}
                    sampleMessages={selectedMbti.sampleMessages}
                    allMessages={selectedMbti.allMessages}
                    onClose={handleClosePopup}
                />
            )}
        </div>
    );
};

const getMbtiColor = (mbti) => {
    const colors = {
        'ISTJ': 'bg-blue-100 text-blue-700', 'ISFJ': 'bg-blue-100 text-blue-700',
        'INFJ': 'bg-green-100 text-green-700', 'INTJ': 'bg-purple-100 text-purple-700',
        'ISTP': 'bg-yellow-100 text-yellow-700', 'ISFP': 'bg-yellow-100 text-yellow-700',
        'INFP': 'bg-green-100 text-green-700', 'INTP': 'bg-purple-100 text-purple-700',
        'ESTP': 'bg-red-100 text-red-700', 'ESFP': 'bg-red-100 text-red-700',
        'ENFP': 'bg-green-100 text-green-700', 'ENTP': 'bg-purple-100 text-purple-700',
        'ESTJ': 'bg-blue-100 text-blue-700', 'ESFJ': 'bg-blue-100 text-blue-700',
        'ENFJ': 'bg-green-100 text-green-700', 'ENTJ': 'bg-purple-100 text-purple-700',
    };
    return colors[mbti] || 'bg-gray-100 text-gray-700';
};

const getMbtiTitle = (mbti) => {
    const titles = {
        'INTJ': 'Architect', 'INTP': 'Logician', 'ENTJ': 'Commander', 'ENTP': 'Debater',
        'INFJ': 'Advocate', 'INFP': 'Mediator', 'ENFJ': 'Protagonist', 'ENFP': 'Campaigner',
        'ISTJ': 'Logistician', 'ISFJ': 'Defender', 'ESTJ': 'Executive', 'ESFJ': 'Consul',
        'ISTP': 'Virtuoso', 'ISFP': 'Adventurer', 'ESTP': 'Entrepreneur', 'ESFP': 'Entertainer'
    };
    return titles[mbti] || mbti;
}

const getMbtiIcon = (mbti) => {
    if (mbti.startsWith('I')) return '🧘';
    if (mbti.startsWith('E')) return '🗣️';
    return '👤';
}

export default TimelinePanel;
