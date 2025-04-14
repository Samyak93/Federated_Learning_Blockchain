// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract GlobalModelStorage {
    // Arrays for global model weights and corresponding sample counts.
    int256[] public globalWeights;
    uint256[] public sampleCounts;

    // Initialize arrays in the constructor.
    constructor(uint256 modelSize, uint256 sampleSize) {
        globalWeights = new int256[](modelSize);
        sampleCounts = new uint256[](sampleSize);
    }

    // Update function that accepts both weights and sample counts.
    function updateGlobalModel(int256[] memory newModel, uint256[] memory newSampleCounts) public {
        require(newModel.length == globalWeights.length, "Model size mismatch");
        require(newSampleCounts.length == sampleCounts.length, "Sample counts size mismatch");
        for (uint i = 0; i < newModel.length; i++) {
            globalWeights[i] = newModel[i];
        }
        for (uint i = 0; i < newSampleCounts.length; i++) {
            sampleCounts[i] = newSampleCounts[i];
        }
    }

    // Getter that returns both arrays.
    function getGlobalModel() public view returns (int256[] memory, uint256[] memory) {
        return (globalWeights, sampleCounts);
    }
}
