import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "rtmlib-ts Playground",
  description: "Test Object Detection and Pose Estimation",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
