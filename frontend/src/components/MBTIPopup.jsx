import React from 'react';
import { mbtiData } from '../utils/mbtiData';

const MBTIPopup = ({ mbti, gender, sampleMessages = [], allMessages = [], onClose }) => {
    const data = mbtiData[mbti];
    const [showMessages, setShowMessages] = React.useState(false);

    if (!data) return null;

    const imageFilename = gender === 'male' ? 'm' : 'f';
    const imagePath = `/16pf/${data.folder}/${imageFilename}.png`;

    return (
        <div className="fixed inset-0 bg-transparent flex justify-center items-center p-4 z-[60] animate-fade-in">
            <div
                className="rounded-2xl shadow-2xl w-full max-w-5xl h-[90vh] overflow-hidden animate-scale-up relative flex flex-col items-center justify-center transition-colors duration-500"
                style={{ backgroundColor: data.color }}
            >
                {!showMessages && (
                    <button
                        onClick={onClose}
                        className="absolute top-6 right-6 w-10 h-10 rounded-full bg-black/10 hover:bg-black/20 flex items-center justify-center text-gray-700 transition-colors z-20"
                    >
                        ✕
                    </button>
                )}

                {!showMessages ? (
                    <>
                        <div className="p-8 flex flex-col items-center text-center max-w-2xl w-full">
                            <h2 className="text-3xl font-bold text-gray-800 mb-4">{data.role}</h2>

                            <div className="w-64 h-64 mb-6 relative">
                                <img
                                    src={imagePath}
                                    alt={`${data.title} character`}
                                    className="w-full h-full object-contain"
                                    onError={(e) => {
                                        e.target.onerror = null;
                                        e.target.src = 'https://via.placeholder.com/300?text=Character+Not+Found';
                                    }}
                                />
                            </div>

                            <h1 className="text-5xl font-bold mb-2" style={{ color: data.textColor }}>The {data.title}</h1>
                            <div className="text-xl text-gray-700 font-semibold mb-6">{mbti}</div>

                            <p className="text-gray-700 text-lg leading-relaxed mb-6">
                                {data.description}
                            </p>

                            {/* Sample Messages */}
                            {sampleMessages.length > 0 && (
                                <div className="w-full bg-white/30 rounded-xl p-4 backdrop-blur-sm">
                                    <p className="text-sm font-semibold text-gray-600 mb-2 text-left">Recent Activity:</p>
                                    <div className="space-y-2 text-left">
                                        {sampleMessages.map((msg, idx) => (
                                            <div key={idx} className="text-gray-800 text-sm truncate opacity-80">
                                                "{msg}"
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}
                        </div>

                        <button
                            onClick={() => setShowMessages(true)}
                            className="absolute bottom-8 right-8 px-6 py-3 bg-white/80 hover:bg-white text-gray-800 font-bold rounded-full shadow-lg transition-all flex items-center gap-2 backdrop-blur-md z-10"
                        >
                            <span>💬</span> Messages
                        </button>
                    </>
                ) : (
                    <div className="w-full h-full flex flex-col p-8 animate-fade-in">
                        <div className="flex items-center justify-between mb-6">
                            <h2 className="text-3xl font-bold text-gray-800">Messages Analysis</h2>
                            <button
                                onClick={() => setShowMessages(false)}
                                className="px-4 py-2 bg-black/5 hover:bg-black/10 rounded-lg text-gray-700 font-medium transition-colors"
                            >
                                ← Back to Profile
                            </button>
                        </div>

                        <div className="flex-1 overflow-y-auto pr-2 custom-scrollbar">
                            {allMessages.length > 0 ? (
                                <div className="space-y-4">
                                    {allMessages.map((msg, idx) => (
                                        <div key={idx} className="bg-white/40 p-4 rounded-xl backdrop-blur-sm border border-white/20">
                                            <p className="text-gray-800 leading-relaxed">"{msg}"</p>
                                        </div>
                                    ))}
                                </div>
                            ) : (
                                <div className="h-full flex items-center justify-center text-gray-500">
                                    No messages found for this personality type.
                                </div>
                            )}
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};

export default MBTIPopup;
