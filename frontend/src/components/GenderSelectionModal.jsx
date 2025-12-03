import React from 'react';

const GenderSelectionModal = ({ onSelect, onClose }) => {
    return (
        <div className="fixed inset-0 bg-transparent backdrop-blur-sm flex justify-center items-center p-4 z-[60] animate-fade-in">
            <div className="bg-white rounded-2xl shadow-2xl p-8 w-full max-w-md animate-scale-up text-center">
                <h2 className="text-2xl font-bold text-gray-800 mb-2">Select Character Gender</h2>
                <p className="text-gray-500 mb-8">Choose a gender to display the appropriate character illustrations.</p>

                <div className="grid grid-cols-2 gap-4">
                    <button
                        onClick={() => onSelect('male')}
                        className="p-6 rounded-xl border-2 border-gray-100 hover:border-blue-500 hover:bg-blue-50 transition-all group"
                    >
                        <div className="text-4xl mb-3 group-hover:scale-110 transition-transform">👨</div>
                        <div className="font-bold text-gray-700 group-hover:text-blue-600">Male</div>
                    </button>

                    <button
                        onClick={() => onSelect('female')}
                        className="p-6 rounded-xl border-2 border-gray-100 hover:border-pink-500 hover:bg-pink-50 transition-all group"
                    >
                        <div className="text-4xl mb-3 group-hover:scale-110 transition-transform">👩</div>
                        <div className="font-bold text-gray-700 group-hover:text-pink-600">Female</div>
                    </button>
                </div>

                <button
                    onClick={onClose}
                    className="mt-6 text-gray-400 hover:text-gray-600 text-sm underline"
                >
                    Cancel
                </button>
            </div>
        </div>
    );
};

export default GenderSelectionModal;
