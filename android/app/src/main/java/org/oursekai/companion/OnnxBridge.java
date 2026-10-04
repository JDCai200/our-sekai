package org.oursekai.companion;

import ai.onnxruntime.*;
import android.util.Base64;
import org.json.*;
import java.nio.*;
import java.util.*;

/** Named-tensor CPU inference. Every input/output has explicit dtype and shape. */
public final class OnnxBridge implements AutoCloseable {
    private final OrtEnvironment environment = OrtEnvironment.getEnvironment();
    private final Map<String, OrtSession> sessions = new HashMap<>();

    public String runJson(String path, String payload) throws Exception {
        OrtSession session = sessions.get(path);
        if (session == null) {
            try (OrtSession.SessionOptions options = new OrtSession.SessionOptions()) {
                options.setIntraOpNumThreads(2);
                options.setInterOpNumThreads(1);
                session = environment.createSession(path, options);
            }
            sessions.put(path, session);
        }
        Map<String, OnnxTensor> inputs = new LinkedHashMap<>();
        try {
            JSONArray rows = new JSONArray(payload);
            for (int index = 0; index < rows.length(); index++) {
                JSONObject row = rows.getJSONObject(index);
                JSONArray dimensions = row.getJSONArray("shape");
                long[] shape = new long[dimensions.length()];
                for (int dimension = 0; dimension < shape.length; dimension++) shape[dimension] = dimensions.getLong(dimension);
                byte[] bytes = Base64.decode(row.getString("data"), Base64.NO_WRAP);
                ByteBuffer buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
                OnnxTensor tensor = row.getBoolean("int64")
                    ? OnnxTensor.createTensor(environment, buffer.asLongBuffer(), shape)
                    : OnnxTensor.createTensor(environment, buffer.asFloatBuffer(), shape);
                inputs.put(row.getString("name"), tensor);
            }
            JSONArray output = new JSONArray();
            try (OrtSession.Result result = session.run(inputs)) {
                for (int index = 0; index < result.size(); index++) {
                    OnnxTensor tensor = (OnnxTensor) result.get(index);
                    FloatBuffer floats = tensor.getFloatBuffer();
                    float[] values = new float[floats.remaining()];
                    floats.get(values);
                    ByteBuffer bytes = ByteBuffer.allocate(values.length * 4).order(ByteOrder.LITTLE_ENDIAN);
                    bytes.asFloatBuffer().put(values);
                    JSONArray shape = new JSONArray();
                    for (long dimension : tensor.getInfo().getShape()) shape.put(dimension);
                    output.put(new JSONObject().put("shape", shape).put("data", Base64.encodeToString(bytes.array(), Base64.NO_WRAP)));
                }
            }
            return output.toString();
        } finally {
            for (OnnxTensor tensor : inputs.values()) tensor.close();
        }
    }

    public void release(String path) throws OrtException {
        OrtSession session = sessions.remove(path);
        if (session != null) session.close();
    }

    public void close() throws OrtException {
        for (OrtSession session : sessions.values()) session.close();
        sessions.clear();
    }
}
