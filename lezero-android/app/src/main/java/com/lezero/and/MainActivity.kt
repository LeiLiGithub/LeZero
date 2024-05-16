package com.lezero.and

import android.os.Bundle
import android.widget.Button
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.chaquo.python.PyObject
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform

class MainActivity: AppCompatActivity() {
    private lateinit var mBtnClear: Button
    private lateinit var mBtnGetData: Button
    private lateinit var mDoodleView: DoodleView

    private lateinit var mPyModule: PyObject

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        initPy()
        mDoodleView = findViewById(R.id.doodle_view)
        mBtnClear = findViewById<Button>(R.id.btn_1).apply {
            setOnClickListener {
                mDoodleView.clearDoodle()
            }
        }
        mBtnGetData = findViewById<Button>(R.id.btn_2).apply {
            setOnClickListener {
                val inputArray = mDoodleView.getData(28, 28)
                val result = mPyModule.callAttr("infer_user_input", inputArray).toInt()
                Toast.makeText(this@MainActivity, "result=$result", Toast.LENGTH_SHORT).show()
//                mPyModule.callAttr("run_train_infer", inputArray)
//                testPy2()
            }
        }
    }

    private fun initPy() {
        if (!Python.isStarted()) {
            Python.start(AndroidPlatform(this))
        }
        val py = Python.getInstance()
        mPyModule = py.getModule("lezero.inference_demo") // 文件名
    }

    private fun testPy() {
        val result = mPyModule.callAttr("sum", 1, 2).toInt()
        Toast.makeText(this, "result=$result", Toast.LENGTH_LONG).show()
    }

    private fun testPy2() {
        mPyModule.callAttr("run_inference", 415)
    }
}
