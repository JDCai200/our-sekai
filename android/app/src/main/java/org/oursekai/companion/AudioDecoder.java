package org.oursekai.companion;

import android.media.*;
import java.io.*;
import java.nio.*;

/** Platform codec -> PCM16 WAV. Python performs crop, resampling and padding. */
public final class AudioDecoder {
    private static void littleInt(RandomAccessFile file, int value) throws IOException {
        file.writeInt(Integer.reverseBytes(value));
    }
    private static void littleShort(RandomAccessFile file, int value) throws IOException {
        file.writeShort(Short.reverseBytes((short) value));
    }
    private static void header(RandomAccessFile output, int rate, int channels) throws IOException {
        long data = output.length() - 44;
        if (data > Integer.MAX_VALUE - 36) throw new IOException("音频超过 WAV 文件尺寸限制");
        output.seek(0); output.writeBytes("RIFF"); littleInt(output, (int) data + 36);
        output.writeBytes("WAVEfmt "); littleInt(output, 16); littleShort(output, 1); littleShort(output, channels);
        littleInt(output, rate); littleInt(output, rate * channels * 2); littleShort(output, channels * 2); littleShort(output, 16);
        output.writeBytes("data"); littleInt(output, (int) data);
    }

    public static void decode(String input, File target, File cancel) throws Exception {
        MediaExtractor extractor = new MediaExtractor();
        MediaCodec codec = null;
        try (RandomAccessFile output = new RandomAccessFile(target, "rw")) {
            output.setLength(0); output.write(new byte[44]);
            extractor.setDataSource(input);
            int track = -1;
            MediaFormat format = null;
            for (int index = 0; index < extractor.getTrackCount(); index++) {
                MediaFormat candidate = extractor.getTrackFormat(index);
                if (candidate.getString(MediaFormat.KEY_MIME).startsWith("audio/")) { track = index; format = candidate; break; }
            }
            if (track == -1) throw new IOException("文件中没有音频轨道");
            extractor.selectTrack(track);
            int rate = format.getInteger(MediaFormat.KEY_SAMPLE_RATE);
            int channels = format.getInteger(MediaFormat.KEY_CHANNEL_COUNT);
            String mime = format.getString(MediaFormat.KEY_MIME);
            if (mime.equals("audio/raw")) {
                ByteBuffer buffer = ByteBuffer.allocate(1024 * 1024);
                while (true) {
                    if (cancel.exists()) throw new InterruptedException("已取消生成");
                    buffer.clear(); int size = extractor.readSampleData(buffer, 0);
                    if (size < 0) break;
                    byte[] bytes = new byte[size]; buffer.position(0); buffer.get(bytes); output.write(bytes); extractor.advance();
                }
            } else {
                codec = MediaCodec.createDecoderByType(mime); codec.configure(format, null, null, 0); codec.start();
                MediaCodec.BufferInfo info = new MediaCodec.BufferInfo();
                boolean inputEnded = false, outputEnded = false;
                int encoding = AudioFormat.ENCODING_PCM_16BIT;
                while (!outputEnded) {
                    if (cancel.exists()) throw new InterruptedException("已取消生成");
                    if (!inputEnded) {
                        int index = codec.dequeueInputBuffer(10000);
                        if (index >= 0) {
                            ByteBuffer buffer = codec.getInputBuffer(index); buffer.clear();
                            int size = extractor.readSampleData(buffer, 0);
                            if (size < 0) { codec.queueInputBuffer(index, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM); inputEnded = true; }
                            else { codec.queueInputBuffer(index, 0, size, extractor.getSampleTime(), 0); extractor.advance(); }
                        }
                    }
                    int index = codec.dequeueOutputBuffer(info, 10000);
                    if (index == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                        MediaFormat decoded = codec.getOutputFormat();
                        rate = decoded.getInteger(MediaFormat.KEY_SAMPLE_RATE); channels = decoded.getInteger(MediaFormat.KEY_CHANNEL_COUNT);
                        if (decoded.containsKey(MediaFormat.KEY_PCM_ENCODING)) encoding = decoded.getInteger(MediaFormat.KEY_PCM_ENCODING);
                    } else if (index >= 0) {
                        ByteBuffer buffer = codec.getOutputBuffer(index);
                        buffer.position(info.offset); buffer.limit(info.offset + info.size);
                        if (encoding == AudioFormat.ENCODING_PCM_FLOAT) {
                            buffer.order(ByteOrder.LITTLE_ENDIAN);
                            ByteBuffer pcm = ByteBuffer.allocate(info.size / 2).order(ByteOrder.LITTLE_ENDIAN);
                            while (buffer.remaining() >= 4) pcm.putShort((short)Math.max(-32768,Math.min(32767,Math.round(buffer.getFloat()*32768))));
                            output.write(pcm.array());
                        } else if (encoding == AudioFormat.ENCODING_PCM_16BIT) {
                            byte[] bytes = new byte[info.size]; buffer.get(bytes); output.write(bytes);
                        } else throw new IOException("设备解码输出不是受支持的 PCM16/Float 格式");
                        outputEnded = (info.flags & MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0;
                        codec.releaseOutputBuffer(index, false);
                    }
                }
            }
            header(output, rate, channels);
        } finally {
            if (codec != null) { try { codec.stop(); } finally { codec.release(); } }
            extractor.release();
        }
    }
}
