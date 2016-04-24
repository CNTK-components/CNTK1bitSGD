#
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
#

class StreamDescription:
    # This class describes a particular stream: its name, element type, storage, etc.

    name          # type: str
    id            # type int
    storage_type  # type: StorageType
    element_type  # type: DataType
    sample_layout # type: NDShape

class MinibatchSource:
    def get_stream_descriptions() -> Set[StreamDescription]:
        # Describes the streams this reader produces.
        raise NotImplementedError()

    def get_next_minibatch(minibatchData: Dict[StreamDescription, Tuple[int, Value]]) -> bool:
        # Reads a minibatch that contains data across all streams.
        # The minibatchData argument specifies the desired minibatch size for each stream of the reader and the actual returned size is the min across all streams
        # The return value of false indciates that the reader will no longer return any further data
        # TODO: Distributed reading support
        raise NotImplementedError()

    def reset_position(newPosition: int):
        # Positions the reader stream to the specified position on the global timeline
        raise NotImplementedError()

    # TODO: Methods to save and restore from checkpoints

def get_stream_description(reader: MinibatchSource, streamName: str) -> StreamDescription:
    # Helper method to get the stream description for the first stream matching the specified 
    pass

def text_minibatch_source(configurationParameters: Dict[str, str]) -> MinibatchSource:
    # Methods to instantiate CNTK built-in MinibatchSources 
    pass
