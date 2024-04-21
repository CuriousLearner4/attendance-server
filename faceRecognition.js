const express = require('express');
const multer = require('multer');
const fs = require('fs');
const path = require('path');
const faceapi = require('face-api.js');
const canvas = require('canvas');
const { Canvas, Image, ImageData } = canvas;
const mongoose = require('mongoose');

faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

const app = express();
const upload = multer({ dest: 'uploads/' });

// Replace with your MongoDB connection string
const uri = 'mongodb+srv://harsha:harsha122@cluster0.srg2hn4.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0';

mongoose.connect(uri, { useNewUrlParser: true, useUnifiedTopology: true })
    .then(() => console.log('Connected to MongoDB'))
    .catch(err => console.error('Error connecting to MongoDB:', err));

const Schema = mongoose.Schema;
const attendanceSchema = new Schema({
    name: { type: String, required: true },
    timestamp: { type: Date, default: Date.now }
});

const Attendance = mongoose.model('Attendance', attendanceSchema);

let faceMatcher;

async function trainModel() {
    await faceapi.nets.ssdMobilenetv1.loadFromDisk('./models');
    await faceapi.nets.faceLandmark68Net.loadFromDisk('./models');
    await faceapi.nets.faceRecognitionNet.loadFromDisk('./models');

    const labeledDescriptors = await loadLabeledImages();

    faceMatcher = new faceapi.FaceMatcher(labeledDescriptors, 0.6);
}

async function loadLabeledImages() {
    const labels = ['13', '30']; // Change to your labels
    const descriptorsPath = path.join(__dirname, 'descriptors.json');

    if (fs.existsSync(descriptorsPath)) {
        console.log('Loading descriptors from file...');
        const descriptorsJson = fs.readFileSync(descriptorsPath, 'utf8');
        const descriptorsData = JSON.parse(descriptorsJson);

        return descriptorsData.map(data => {
            return new faceapi.LabeledFaceDescriptors(data.label, data.descriptors.map(d => new Float32Array(d)));
        });
    } else {
        console.log('Computing descriptors...');

        const labeledDescriptors = await Promise.all(
            labels.map(async (label) => {
                const descriptors = [];
                const imgDir = path.join(__dirname, 'images', label);
                const imgFiles = fs.readdirSync(imgDir);

                for (const imgFile of imgFiles) {
                    const imgPath = path.join(imgDir, imgFile);
                    const imgBuff = fs.readFileSync(imgPath);
                    const img = await canvas.loadImage(imgBuff);
                    const detections = await faceapi.detectSingleFace(img).withFaceLandmarks().withFaceDescriptor();
                    descriptors.push(Array.from(detections.descriptor));
                }

                return { label, descriptors };
            })
        );

        fs.writeFileSync(descriptorsPath, JSON.stringify(labeledDescriptors, null, 2));

        return labeledDescriptors.map(data => {
            return new faceapi.LabeledFaceDescriptors(data.label, data.descriptors.map(d => new Float32Array(d)));
        });
    }
}

app.post('/verify', upload.single('image'), async (req, res) => {
    try {
        const imgBuff = fs.readFileSync(req.file.path);
        const img = await canvas.loadImage(imgBuff);
        const detections = await faceapi.detectSingleFace(img).withFaceLandmarks().withFaceDescriptor();
        
        if (!detections) {
            res.status(400).json({ error: 'No face detected in the image' });
            return;
        }

        const result = faceMatcher.findBestMatch(detections.descriptor);

        if (result.label === 'unknown') {
            res.status(404).json({ error: 'Student does not belong to this class' });
            return;
        }

        const attendance = new Attendance({
            name: result.label
        });

        await attendance.save();
        console.log(`Attendance marked for ${result.label}`);

        res.json({ label: result.label, confidence: result.distance.toFixed(2) });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

app.listen(3000, () => console.log('Server started on port 3000'));

trainModel().catch(console.error);
