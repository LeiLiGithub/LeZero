package com.lezero.and

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Path
import android.util.AttributeSet
import android.util.Log
import android.view.MotionEvent
import android.view.View
import java.lang.StringBuilder
import kotlin.random.Random

/**
 * 涂鸦View，可生成Bitmap
 */
class DoodleView(context: Context, attrs: AttributeSet): View(context, attrs) {
    private val TAG = "DoodleView"

    private val paint = Paint()
    private val path = Path()

    init {
        // 背景色
        setBackgroundColor(Color.BLACK)

        // 画笔色
        paint.color = Color.WHITE
        paint.strokeWidth = 50f
        paint.style = Paint.Style.STROKE
        paint.strokeCap = Paint.Cap.ROUND
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        canvas.drawPath(path, paint)
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        val x = event.x
        val y = event.y

        when (event.action) {
            MotionEvent.ACTION_DOWN -> {
                path.moveTo(x, y)
            }
            MotionEvent.ACTION_MOVE -> {
                path.lineTo(x, y)
            }
        }
        invalidate()
        return true
    }

    /**
     * 将Bitmap生成byte数组
     */
    fun getData(width: Int, height: Int): IntArray {
        val result = IntArray(width * height)
        try {
            isDrawingCacheEnabled = true // 输出到Bitmap
            drawingCacheQuality = View.DRAWING_CACHE_QUALITY_LOW
            val cache = drawingCache
            fillBitmapData(cache, result, width, height)
        } finally {
            isDrawingCacheEnabled = false
        }
        return result
    }

    // 从Bitmap生成IntArray, 写入到参数array中
    private fun fillBitmapData(rawBitmap: Bitmap, array: IntArray, smallWidth: Int, smallHeight: Int) {
        // 缩放比例
        val scaleWidth = smallWidth.toFloat() / rawBitmap.width
        val scaleHeight = smallHeight.toFloat() / rawBitmap.height
        // 缩放matrix
        val matrix = Matrix().apply {
            postScale(scaleWidth, scaleHeight)
        }
        val smallBitmap = Bitmap.createBitmap(rawBitmap, 0, 0, rawBitmap.width, rawBitmap.height, matrix, true)
        // 将bitmap生成像素图
//        val rand = Random(415)
        for (i in 0 until smallWidth) {
//            val sb = StringBuilder()
            for (j in 0 until smallHeight) { // 注意getPixe入参为横纵坐标，对应的是col、row
//                val isDot = (if (smallBitmap.getPixel(j, i) == Color.BLACK) 0 else 1)
                val isDot = (
                        if (smallBitmap.getPixel(j, i) == Color.BLACK)
                            0
                        else
                            255
                )
                array[smallWidth * i + j] = isDot
//                sb.append(if (isDot > 0) "*" else " ")
            }
//            Log.e(TAG, sb.toString())
//            sb.clear()
        }
    }

    fun clearDoodle() {
        path.reset()
        invalidate()
    }
}