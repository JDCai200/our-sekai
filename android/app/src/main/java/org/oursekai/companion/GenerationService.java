package org.oursekai.companion;

import android.app.*;
import android.content.*;
import android.os.*;
import com.chaquo.python.*;
import com.chaquo.python.android.AndroidPlatform;
import org.json.*;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.UUID;
import java.security.MessageDigest;

/** Foreground generation survives leaving the app. No network permission. */
public final class GenerationService extends Service {
    public static volatile boolean running = false;
    private volatile File job;
    private PowerManager.WakeLock wake;

    public static synchronized Python python(Context context) {
        if (!Python.isStarted()) Python.start(new AndroidPlatform(context.getApplicationContext()));
        return Python.getInstance();
    }

    private void write(File file, String value) throws IOException {
        try (FileOutputStream output = new FileOutputStream(file)) { output.write(value.getBytes(StandardCharsets.UTF_8)); }
    }

    private void notification() {
        String channel = "generation";
        if (Build.VERSION.SDK_INT >= 26) {
            NotificationManager manager = getSystemService(NotificationManager.class);
            manager.createNotificationChannel(new NotificationChannel(channel,"本地谱面生成",NotificationManager.IMPORTANCE_LOW));
        }
        Intent cancel = new Intent(this,GenerationService.class).setAction("cancel");
        PendingIntent action = PendingIntent.getService(this,1,cancel,PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE);
        PendingIntent open = PendingIntent.getActivity(this,0,new Intent(this,MainActivity.class),PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE);
        Notification.Builder builder = Build.VERSION.SDK_INT >= 26 ? new Notification.Builder(this,channel) : new Notification.Builder(this);
        startForeground(1,builder.setSmallIcon(android.R.drawable.ic_media_play).setContentTitle("Our Sekai 正在本地生成谱面")
            .setContentText("无需网络；可以返回应用查看进度或取消").setContentIntent(open).setOngoing(true)
            .addAction(android.R.drawable.ic_menu_close_clear_cancel,"取消",action).build());
    }

    @Override public int onStartCommand(Intent intent,int flags,int startId) {
        if (intent == null) { stopSelf(); return START_NOT_STICKY; }
        if ("cancel".equals(intent.getAction())) {
            if (job != null) try { new File(job,"cancel").createNewFile(); } catch (IOException ignored) {}
            if (!running) stopSelf();
            return START_NOT_STICKY;
        }
        if (running) return START_NOT_STICKY;
        running = true;
        notification();
        PowerManager manager = getSystemService(PowerManager.class);
        wake = manager.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK,"OurSekai:Generation");
        wake.acquire(30*60*1000L);
        String request = intent.getStringExtra("request");
        job = new File(getFilesDir(),"Jobs/"+UUID.randomUUID()); job.mkdirs();
        getSharedPreferences("state",MODE_PRIVATE).edit().putString("job",job.getAbsolutePath()).apply();
        new Thread(() -> generate(request),"OurSekai-Generation").start();
        return START_NOT_STICKY;
    }

    private void generate(String requestText) {
        try {
            write(new File(job,"status.json"),new JSONObject().put("message","正在准备本地模型和音频").put("percent",1).toString());
            File models = new File(getFilesDir(),"Models"); models.mkdirs();
            ByteArrayOutputStream registryBytes=new ByteArrayOutputStream();
            try (InputStream input=getAssets().open("models/model-lock.json")) {
                byte[] buffer=new byte[4096]; int size;
                while ((size=input.read(buffer))!=-1) registryBytes.write(buffer,0,size);
            }
            JSONArray registry=new JSONObject(registryBytes.toString("UTF-8")).getJSONArray("files");
            java.util.Map<String,JSONObject> expected=new java.util.HashMap<>();
            for (int i=0;i<registry.length();i++) expected.put(registry.getJSONObject(i).getString("path"),registry.getJSONObject(i));
            for (String name : getAssets().list("models")) {
                File target = new File(models,name);
                boolean valid=false;
                JSONObject row=expected.get(name);
                if (row!=null && target.isFile() && target.length()==row.getLong("size")) {
                    MessageDigest digest=MessageDigest.getInstance("SHA-256");
                    try (InputStream input=new FileInputStream(target)) {
                        byte[] buffer=new byte[1024*1024]; int size;
                        while ((size=input.read(buffer))!=-1) digest.update(buffer,0,size);
                    }
                    StringBuilder hex=new StringBuilder(); for (byte value:digest.digest()) hex.append(String.format("%02x",value & 255));
                    valid=hex.toString().equals(row.getString("sha256"));
                }
                if (!valid) {
                    File temporary = new File(models,name+".tmp");
                    try (InputStream input = getAssets().open("models/"+name); OutputStream output = new FileOutputStream(temporary)) {
                        byte[] buffer = new byte[1024*1024]; int size;
                        while ((size=input.read(buffer))!=-1) output.write(buffer,0,size);
                    }
                    android.system.Os.rename(temporary.getAbsolutePath(),target.getAbsolutePath());
                }
            }
            JSONObject request = new JSONObject(requestText);
            String input = request.getString("audio");
            if (!input.toLowerCase().endsWith(".wav")) {
                File decoded = new File(job,"decoded.wav");
                AudioDecoder.decode(input,decoded,new File(job,"cancel"));
                request.put("audio",decoded.getAbsolutePath());
            }
            String result = python(this).getModule("portable.android_bridge").callAttr("generate",
                request.toString(),models.getAbsolutePath(),job.getAbsolutePath(),getFilesDir().getAbsolutePath()).toString();
            write(new File(job,"result.json"),result);
            JSONObject parsed = new JSONObject(result);
            if (parsed.optBoolean("success")) getSharedPreferences("state",MODE_PRIVATE).edit().putString("zip",parsed.getString("zip")).apply();
        } catch (Exception error) {
            try {
                boolean cancelled = new File(job,"cancel").exists();
                write(new File(job,"result.json"),new JSONObject().put("success",false).put("cancelled",cancelled)
                    .put("message",cancelled ? "已取消生成" : String.valueOf(error.getMessage())).toString());
                try (PrintWriter log = new PrintWriter(new File(job,"native-error.log"))) { error.printStackTrace(log); }
            } catch (Exception ignored) {}
        } finally {
            running = false;
            if (wake != null && wake.isHeld()) wake.release();
            stopForeground(STOP_FOREGROUND_REMOVE);
            stopSelf();
        }
    }

    @Override public IBinder onBind(Intent intent) { return null; }
}
